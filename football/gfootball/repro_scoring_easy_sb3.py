#!/usr/bin/env python3
"""
Equivalente do repro_scoring_easy.sh usando stable-baselines3,
com:
  - curriculum learning adaptativo (progressão baseada em performance)
  - self-play opcional (treinar contra versões anteriores)
  - paralelização otimizada para RTX 4090
  - monitoramento completo (wandb) e logging detalhado de estatísticas de partidas.
"""

import os
import collections
import shutil
import signal
import sys
import traceback
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gym

# ===========================
# Configurações gerais
# ===========================

# Configurações de Hardware (otimizado para RTX 4090)
# RTX 4090 tem 24GB VRAM, pode suportar 32-64 ambientes dependendo da configuração
# Reduzido para 64 para maior estabilidade e evitar SIGSEGV
NUM_ENVS = int(os.environ.get("NUM_ENVS", "64"))  # Reduzido de 96 para 64 para estabilidade

# Configurações de Self-Play (sempre ativado)
ENABLE_SELF_PLAY = True
SELF_PLAY_RATIO = float(os.environ.get("SELF_PLAY_RATIO", "0.5"))  # 50% self-play
SELF_PLAY_STAGES = os.environ.get("SELF_PLAY_STAGES", "").split(",")  # Estágios que usam self-play
SELF_PLAY_POOL_SIZE = int(os.environ.get("SELF_PLAY_POOL_SIZE", "10"))  # Máximo de checkpoints no pool

# Configurações de Curriculum Adaptativo (sempre ativado)
ADAPTIVE_CURRICULUM = True
MIN_WIN_RATE_TO_ADVANCE = float(os.environ.get("MIN_WIN_RATE_TO_ADVANCE", "0.7"))  # 70% vitória
MIN_EPISODES_FOR_EVAL = int(os.environ.get("MIN_EPISODES_FOR_EVAL", "100"))  # Mínimo de episódios
WINDOW_SIZE = int(os.environ.get("CURRICULUM_WINDOW_SIZE", "200"))  # Janela de avaliação

# Novos critérios rigorosos de avanço
MIN_GOAL_DIFFERENCE = int(os.environ.get("MIN_GOAL_DIFFERENCE", "1"))  # Diferença mínima de gols
MIN_WIN_BY_GOAL_DIFFERENCE_RATE = float(os.environ.get("MIN_WIN_BY_GOAL_DIFFERENCE_RATE", "0.9"))  # 90% das vitórias com >= 1 gol
MIN_GOALS_SCORED = float(os.environ.get("MIN_GOALS_SCORED", "1.0"))  # Gols médios marcados
MAX_GOALS_CONCEDED = float(os.environ.get("MAX_GOALS_CONCEDED", "0.5"))  # Gols médios sofridos
STABILITY_EVALUATIONS = int(os.environ.get("STABILITY_EVALUATIONS", "2"))  # Número de avaliações consecutivas necessárias
MAX_TIMESTEPS_PER_STAGE = int(os.environ.get("MAX_TIMESTEPS_PER_STAGE", "10000000"))  # Safety net: 10M timesteps
EVAL_CHUNK_SIZE = int(os.environ.get("EVAL_CHUNK_SIZE", "100000"))  # Verificar a cada 100k timesteps

# Configurações de regressão
REGRESSION_WIN_RATE_THRESHOLD = float(os.environ.get("REGRESSION_WIN_RATE_THRESHOLD", "0.3"))  # 30% para regressão
REGRESSION_EVALUATIONS = int(os.environ.get("REGRESSION_EVALUATIONS", "3"))  # 3 avaliações consecutivas
REGRESSION_COOLDOWN = int(os.environ.get("REGRESSION_COOLDOWN", "5"))  # Cooldown em chunks para evitar oscilação

# Hiperparâmetros otimizados para RTX 4090
# Ajustados para treino ESTÁVEL e RÁPIDO
N_STEPS = 1024  # Mantido alto para melhor estimativa de valor
N_EPOCHS = 4  # REDUZIDO de 8 para 4 - muito epoch com recompensas não normalizadas causa instabilidade
N_MINIBATCHES = 8  # Mantido
LR = 0.00015  # REDUZIDO de 0.0003 para 0.00015 - mais estável com recompensas densas
GAMMA = 0.999  # Mantido alto para valorizar recompensas futuras
ENT_COEF = 0.01  # Mantido para boa exploração
CLIP_RANGE = 0.115  # Mantido
MAX_GRAD_NORM = 0.5  # REDUZIDO de 0.76 para 0.5 - mais conservador, evita updates grandes
GAE_LAMBDA = 0.98  # Mantido
VF_COEF = 0.5  # Mantido

# Checkpointing mais frequente para 24/7 e recovery
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", "50000"))  # A cada 50k steps
RECOVERY_CHECKPOINT_FREQ = int(os.environ.get("RECOVERY_CHECKPOINT_FREQ", "25000"))  # Recovery a cada 25k steps

# Configurações de Reward Shaping
USE_CHECKPOINT_REWARDS = os.environ.get("USE_CHECKPOINT_REWARDS", "true").lower() == "true"
USE_DENSE_REWARDS = os.environ.get("USE_DENSE_REWARDS", "true").lower() == "true"
DENSE_REWARD_SCALE = float(os.environ.get("DENSE_REWARD_SCALE", "1.0"))

# Configuração de Normalização (importante para estabilidade)
USE_VEC_NORMALIZE = os.environ.get("USE_VEC_NORMALIZE", "true").lower() == "true"

# Curriculum: estágios de dificuldade crescente
# Nota: total_timesteps agora é apenas uma estimativa inicial, o treinamento continua até atingir critérios
# Ordem crescente de dificuldade baseada nos cenários disponíveis
CURRICULUM_STAGES = [
    # Estágio 1: Gol vazio (mais fácil)
    {
        "name": "stage1_empty_goal_close",
        "level": "academy_empty_goal_close",
        "total_timesteps": int(2e6),
    },
    {
        "name": "stage1.5_empty_goal",
        "level": "academy_empty_goal",
        "total_timesteps": int(2e6),
    },
    # Estágio 2: Correr para marcar (sem goleiro)
    {
        "name": "stage2_run_to_score",
        "level": "academy_run_to_score",
        "total_timesteps": int(3e6),
    },
    # Estágio 3: Correr para marcar com goleiro
    {
        "name": "stage3_run_to_score_with_keeper",
        "level": "academy_run_to_score_with_keeper",
        "total_timesteps": int(4e6),
    },
    # Estágio 4: Passar e chutar com goleiro
    {
        "name": "stage4_pass_and_shoot_with_keeper",
        "level": "academy_pass_and_shoot_with_keeper",
        "total_timesteps": int(4e6),
    },
    {
        "name": "stage4.5_run_pass_and_shoot_with_keeper",
        "level": "academy_run_pass_and_shoot_with_keeper",
        "total_timesteps": int(4e6),
    },
    # Estágio 5: Situações mais complexas
    {
        "name": "stage5_3_vs_1_with_keeper",
        "level": "academy_3_vs_1_with_keeper",
        "total_timesteps": int(5e6),
    },
    {
        "name": "stage5.5_corner",
        "level": "academy_corner",
        "total_timesteps": int(5e6),
    },
    # Estágio 6: Contra-ataques
    {
        "name": "stage6_counterattack_easy",
        "level": "academy_counterattack_easy",
        "total_timesteps": int(6e6),
    },
    {
        "name": "stage6.5_counterattack_hard",
        "level": "academy_counterattack_hard",
        "total_timesteps": int(6e6),
    },
    # Estágio 7: Jogos pequenos
    {
        "name": "stage7_1v1_easy",
        "level": "1_vs_1_easy",
        "total_timesteps": int(7e6),
    },
    {
        "name": "stage7.5_5v5",
        "level": "5_vs_5",
        "total_timesteps": int(7e6),
    },
    # Estágio 8: Jogos completos (11v11)
    {
        "name": "stage8_single_goal_versus_lazy",
        "level": "academy_single_goal_versus_lazy",
        "total_timesteps": int(8e6),
    },
    {
        "name": "stage8.5_11v11_easy",
        "level": "11_vs_11_easy_stochastic",
        "total_timesteps": int(9e6),
    },
    {
        "name": "stage9_11v11_stochastic",
        "level": "11_vs_11_stochastic",
        "total_timesteps": int(10e6),
    },
    {
        "name": "stage9.5_11v11_hard",
        "level": "11_vs_11_hard_stochastic",
        "total_timesteps": int(11e6),
    },
    {
        "name": "stage10_11v11_competition",
        "level": "11_vs_11_competition",
        "total_timesteps": int(12e6),
    },
]

# Diretórios base (compatíveis com o setup em docker-compose)
LOG_DIR_BASE = "/RL/logs_repro_scoring"
CHECKPOINT_DIR_BASE = "/RL/checkpoints_repro_scoring"
EVAL_DIR_BASE = "/RL/eval_repro_scoring"

os.makedirs(LOG_DIR_BASE, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)
os.makedirs(EVAL_DIR_BASE, exist_ok=True)


class SafeWrapper(gym.Wrapper):
    """Wrapper de segurança que captura erros e previne crashes."""
    
    def __init__(self, env, max_retries=3):
        super().__init__(env)
        self.max_retries = max_retries
        self.reset_count = 0
        self.step_count = 0
        self.error_count = 0
        self.consecutive_reset_errors = 0
        self.max_consecutive_reset_errors = 5
        
    def step(self, action):
        """Step com tratamento de erros robusto."""
        for attempt in range(self.max_retries):
            try:
                obs, reward, done, info = self.env.step(action)
                self.step_count += 1
                self.error_count = 0  # Reset contador de erros em sucessos consecutivos
                return obs, reward, done, info
            except Exception as e:
                self.error_count += 1
                if attempt < self.max_retries - 1:
                    print(f"⚠ Erro no step (tentativa {attempt + 1}/{self.max_retries}): {e}")
                    # Tentar resetar o ambiente antes de retentar
                    try:
                        self.env.reset()
                    except:
                        pass
                else:
                    # Última tentativa falhou, retornar valores seguros
                    print(f"⚠ Erro crítico no step após {self.max_retries} tentativas: {e}")
                    # Retornar valores padrão seguros
                    obs = self.observation_space.sample() if hasattr(self, 'observation_space') else None
                    reward = 0.0
                    done = True  # Forçar término do episódio problemático
                    info = {"error": str(e), "safe_fallback": True}
                    return obs, reward, done, info
        
        # Fallback final
        obs = self.observation_space.sample() if hasattr(self, 'observation_space') else None
        return obs, 0.0, True, {"error": "max_retries_exceeded", "safe_fallback": True}
    
    def reset(self, **kwargs):
        """Reset com tratamento de erros robusto e limpeza de memória."""
        import gc
        import time
        
        for attempt in range(self.max_retries):
            try:
                # Limpar memória antes do reset se muitos erros consecutivos
                if self.consecutive_reset_errors > 2:
                    gc.collect()
                    time.sleep(0.01)  # Pequeno delay para liberar recursos
                
                obs = self.env.reset(**kwargs)
                self.reset_count += 1
                self.consecutive_reset_errors = 0  # Reset contador de erros
                self.error_count = 0
                return obs
            except Exception as e:
                self.consecutive_reset_errors += 1
                error_msg = str(e)
                
                if attempt < self.max_retries - 1:
                    print(f"⚠ Erro no reset (tentativa {attempt + 1}/{self.max_retries}): {error_msg}")
                    
                    # Limpar memória entre tentativas
                    gc.collect()
                    
                    # Delay progressivo: mais tempo entre tentativas
                    time.sleep(0.1 * (attempt + 1))
                    
                    # Tentar fechar e recriar se muitos erros consecutivos
                    if self.consecutive_reset_errors >= self.max_consecutive_reset_errors:
                        print(f"⚠ Muitos erros consecutivos ({self.consecutive_reset_errors}). "
                              f"Tentando limpar recursos...")
                        try:
                            self.env.close()
                            gc.collect()
                            time.sleep(0.5)
                        except:
                            pass
                        # Indica que precisa recriar o ambiente
                        raise RuntimeError(
                            f"Ambiente corrompido após {self.consecutive_reset_errors} erros consecutivos. "
                            f"Precisa ser recriado."
                        )
                else:
                    print(f"⚠ Erro crítico no reset após {self.max_retries} tentativas: {error_msg}")
                    # Última tentativa: limpar tudo e indicar necessidade de recriação
                    try:
                        gc.collect()
                    except:
                        pass
                    raise RuntimeError(
                        f"Não foi possível resetar o ambiente após {self.max_retries} tentativas. "
                        f"Erro: {error_msg}"
                    )
        
        raise RuntimeError("Erro no reset: max_retries excedido")


class ScoreInfoWrapper(gym.Wrapper):
    """Wrapper que adiciona o score ao info dict quando o episódio termina."""
    
    def __init__(self, env):
        super().__init__(env)
        self.last_observation = None
    
    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)
            self.last_observation = obs
            
            # Garantir que info é um dicionário
            if info is None:
                info = {}
            elif not isinstance(info, dict):
                info = {"info": info}
            
            # Quando o episódio termina, adicionar score ao info
            if done:
                # O score está sempre na observação como um dict com chave 'score'
                # que é uma lista [left_goals, right_goals]
                if isinstance(obs, dict) and "score" in obs:
                    score = obs["score"]
                    # Garantir que é uma lista/array
                    if isinstance(score, (list, tuple, np.ndarray)) and len(score) >= 2:
                        info["score"] = [int(score[0]), int(score[1])]
                    else:
                        # Fallback: tentar converter
                        try:
                            info["score"] = [int(score[0]), int(score[1])]
                        except:
                            info["score"] = [0, 0]
                else:
                    # Se não encontrou na observação atual, tentar da última observação
                    if self.last_observation is not None:
                        if isinstance(self.last_observation, dict) and "score" in self.last_observation:
                            score = self.last_observation["score"]
                            if isinstance(score, (list, tuple, np.ndarray)) and len(score) >= 2:
                                info["score"] = [int(score[0]), int(score[1])]
                            else:
                                info["score"] = [0, 0]
                        else:
                            info["score"] = [0, 0]
                    else:
                        info["score"] = [0, 0]
            
            return obs, reward, done, info
        except Exception as e:
            # Capturar qualquer erro e retornar valores seguros
            print(f"⚠ Erro no ScoreInfoWrapper.step: {e}")
            obs = self.observation_space.sample() if hasattr(self, 'observation_space') else self.last_observation
            info = {"score": [0, 0], "error": str(e), "safe_fallback": True}
            return obs, 0.0, True, info
    
    def reset(self, **kwargs):
        try:
            obs = self.env.reset(**kwargs)
            self.last_observation = obs
            return obs
        except Exception as e:
            print(f"⚠ Erro no ScoreInfoWrapper.reset: {e}")
            # Retentar reset
            try:
                obs = self.env.reset(**kwargs)
                self.last_observation = obs
                return obs
            except:
                raise


class DenseRewardWrapper(gym.RewardWrapper):
    """
    Wrapper que adiciona recompensas densas para acelerar aprendizado:
    - Recompensas por posse de bola
    - Recompensas por aproximação do gol
    - Recompensas por chutes (AUMENTADAS para incentivar mais tentativas)
    - Recompensas por controle efetivo
    - Penalidades por perder posse
    """
    
    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        
        # Parâmetros de recompensa (AUMENTADOS para incentivar mais ações ofensivas)
        self.reward_possession = 0.02  # Aumentado de 0.01 para 0.02
        self.reward_approach = 0.05  # Aumentado de 0.02 para 0.05
        self.reward_shot = 0.2  # AUMENTADO de 0.05 para 0.2 - incentiva chutes!
        self.reward_control = 0.01  # Aumentado de 0.005 para 0.01
        self.reward_near_goal = 0.1  # NOVO: recompensa por estar muito perto do gol
        self.penalty_lose_possession = -0.01  # Penalidade por perder posse
        
        # Estado anterior para calcular progresso (limitado para evitar vazamento de memória)
        self.max_state_history = 1000  # Limitar histórico
        self.last_ball_distance = {}
        self.last_had_possession = {}
        self.last_ball_position = {}
        self.call_count = 0
        self.reward_applied = False  # Flag para print único
        
        # Print confirmando inicialização
        if USE_DENSE_REWARDS:
            print("✓ Recompensas densas personalizadas ATIVADAS")
            print(f"  - Recompensa por posse: {self.reward_possession}")
            print(f"  - Recompensa por aproximação: {self.reward_approach}")
            print(f"  - Recompensa por chute: {self.reward_shot} (AUMENTADA para incentivar!)")
            print(f"  - Recompensa perto do gol: {self.reward_near_goal}")
            print(f"  - Escala: {self.reward_scale}")
        
    def reset(self, **kwargs):
        """Reset wrapper state com limpeza de memória."""
        obs = self.env.reset(**kwargs)
        # Limpar todos os estados anteriores para evitar vazamento de memória
        self.last_ball_distance.clear()
        self.last_had_possession.clear()
        self.last_ball_position.clear()
        
        # Limpar memória periodicamente
        if self.call_count % 100 == 0:
            import gc
            gc.collect()
        
        return obs
    
    def reward(self, reward):
        """
        Adiciona recompensas densas baseadas no estado atual.
        reward pode ser um valor único ou lista (ambiente vetorizado).
        """
        if not USE_DENSE_REWARDS:
            return reward
        
        self.call_count += 1
        
        # Print único confirmando aplicação (apenas uma vez)
        if not self.reward_applied and self.call_count == 1:
            print("✓ Recompensas normais + personalizadas DEFINIDAS e sendo aplicadas!")
            self.reward_applied = True
        
        # Limitar tamanho dos dicionários de estado periodicamente
        if self.call_count % 500 == 0:
            # Limpar estados antigos para evitar vazamento
            if len(self.last_ball_distance) > self.max_state_history:
                # Manter apenas os mais recentes
                keys_to_keep = list(self.last_ball_distance.keys())[-self.max_state_history:]
                self.last_ball_distance = {k: self.last_ball_distance[k] for k in keys_to_keep}
                self.last_had_possession = {k: self.last_had_possession.get(k, False) for k in keys_to_keep}
                self.last_ball_position = {k: self.last_ball_position.get(k) for k in keys_to_keep if k in self.last_ball_position}
        
        # Obter observação atual
        try:
            # Tentar acessar observação via unwrapped
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'observation'):
                observation = self.env.unwrapped.observation()
            elif hasattr(self.env, 'observation'):
                observation = self.env.observation()
            else:
                # Se não conseguir acessar observação, retornar reward original
                return reward
        except Exception:
            # Se não conseguir acessar observação, retornar reward original
            return reward
        
        if observation is None:
            return reward
        
        # Processar recompensas (pode ser ambiente único ou vetorizado)
        if isinstance(reward, (list, tuple, np.ndarray)):
            return self._process_vectorized_rewards(reward, observation)
        else:
            return self._process_single_reward(reward, observation, 0)
    
    def _process_vectorized_rewards(self, rewards, observations):
        """Processa recompensas para ambiente vetorizado."""
        if not isinstance(observations, (list, tuple)):
            # Se não é lista, tratar como único ambiente
            if len(rewards) > 0:
                rewards[0] = self._process_single_reward(rewards[0], observations, 0)
            return rewards
        
        # Processar cada ambiente
        for i in range(min(len(rewards), len(observations))):
            rewards[i] = self._process_single_reward(rewards[i], observations[i], i)
        
        return rewards
    
    def _process_single_reward(self, base_reward, obs, env_idx):
        """Processa recompensas para um único ambiente."""
        if not isinstance(obs, dict):
            return base_reward
        
        dense_reward = 0.0
        
        # 1. Recompensa por posse de bola
        has_possession = (
            obs.get('ball_owned_team', -1) == 0 and  # Time esquerdo (agente)
            obs.get('ball_owned_player', -1) == obs.get('active', -1)  # Jogador ativo
        )
        
        if has_possession:
            dense_reward += self.reward_possession * self.reward_scale
            # Recompensa adicional por controle efetivo
            dense_reward += self.reward_control * self.reward_scale
            
            # 2. Recompensa por aproximação do gol (AUMENTADA para incentivar mais)
            if 'ball' in obs and len(obs['ball']) >= 2:
                ball_x, ball_y = obs['ball'][0], obs['ball'][1]
                # Gol adversário está em x=1, y=0
                distance_to_goal = np.sqrt((ball_x - 1.0)**2 + (ball_y - 0.0)**2)
                
                # Recompensa por estar mais perto do gol (VALORES AUMENTADOS)
                if distance_to_goal < 0.3:  # MUITO perto do gol - recompensa alta!
                    dense_reward += self.reward_near_goal * 3 * self.reward_scale  # +0.3
                    dense_reward += self.reward_approach * 3 * self.reward_scale  # +0.15
                elif distance_to_goal < 0.5:  # Muito perto do gol
                    dense_reward += self.reward_near_goal * 2 * self.reward_scale  # +0.2
                    dense_reward += self.reward_approach * 2 * self.reward_scale  # +0.1
                elif distance_to_goal < 0.7:  # Perto do gol
                    dense_reward += self.reward_near_goal * self.reward_scale  # +0.1
                    dense_reward += self.reward_approach * self.reward_scale  # +0.05
                
                # Recompensa por progresso (redução de distância) - ESTABILIZADO
                if self.last_ball_distance.get(env_idx) is not None:
                    distance_reduction = self.last_ball_distance[env_idx] - distance_to_goal
                    if distance_reduction > 0:  # Aproximou do gol
                        # Recompensa proporcional ao progresso (REDUZIDO de 20 para 5 para estabilidade)
                        progress_reward = self.reward_approach * distance_reduction * 5 * self.reward_scale
                        # Limitar progress reward para evitar valores extremos
                        progress_reward = np.clip(progress_reward, 0.0, 0.1)  # Máximo 0.1
                        dense_reward += progress_reward
                
                self.last_ball_distance[env_idx] = distance_to_goal
                self.last_ball_position[env_idx] = (ball_x, ball_y)
        
        # 3. Penalidade por perder posse
        if self.last_had_possession.get(env_idx, False) and not has_possession:
            dense_reward += self.penalty_lose_possession * self.reward_scale
        
        self.last_had_possession[env_idx] = has_possession
        
        # 4. Detectar chutes e ações ofensivas (AUMENTADO significativamente)
        if has_possession and 'ball' in obs and len(obs['ball']) >= 2:
            ball_x = obs['ball'][0]
            
            # Recompensa AGGRESSIVA por estar na área do gol
            if ball_x > 0.85:  # Área do gol
                dense_reward += self.reward_shot * 2 * self.reward_scale  # +0.4 por estar na área!
                
                # Se bola entrou na área (estava mais longe antes)
                if (self.last_ball_position.get(env_idx) is not None and
                    self.last_ball_position[env_idx][0] < 0.85):
                    # Bola entrou na área - possível chute - RECOMPENSA ALTA!
                    dense_reward += self.reward_shot * 3 * self.reward_scale  # +0.6 por entrar na área!
            
            # Recompensa adicional por bola avançando em direção ao gol
            if (self.last_ball_position.get(env_idx) is not None):
                last_x = self.last_ball_position[env_idx][0]
                if ball_x > last_x and ball_x > 0.7:  # Avançando e já perto do gol
                    dense_reward += self.reward_approach * 2 * self.reward_scale  # +0.1 por avançar
        
        # NORMALIZAÇÃO E LIMITAÇÃO DE RECOMPENSAS DENSAS para estabilidade
        # Limitar recompensas densas para evitar valores extremos que causam instabilidade
        dense_reward = np.clip(dense_reward, -0.3, 1.5)  # Limitar range: mínimo -0.3, máximo 1.5
        
        # Escalar recompensas densas para não dominarem a recompensa de gol (+1.0)
        # Reduzir escala final das recompensas densas para manter proporção
        dense_reward_scaled = dense_reward * 0.7  # Reduzir para 70% para não dominar
        
        return base_reward + dense_reward_scaled


def make_env(level: str, log_dir: str):
    """Cria ambiente do gfootball para um dado nível (cenário) com wrappers de segurança."""
    try:
        # Configurar tipo de recompensas
        rewards_config = "scoring"
        if USE_CHECKPOINT_REWARDS:
            rewards_config = "scoring,checkpoints"
        
        env = football_env.create_environment(
            env_name=level,
            stacked=True,
            representation="extracted",
            rewards=rewards_config,
            logdir=log_dir,
            render=False,
        )
        # Envolver com wrapper de segurança primeiro (mais próximo do env original)
        env = SafeWrapper(env, max_retries=3)
        # Depois adicionar wrapper que adiciona score ao info
        env = ScoreInfoWrapper(env)
        # Adicionar wrapper de recompensas densas se habilitado
        if USE_DENSE_REWARDS:
            env = DenseRewardWrapper(env, reward_scale=DENSE_REWARD_SCALE)
        return env
    except Exception as e:
        print(f"⚠ Erro ao criar ambiente {level}: {e}")
        print(f"  Logdir: {log_dir}")
        traceback.print_exc()
        raise


class MatchStatsCallback(BaseCallback):
    """
    Callback expandido para capturar estatísticas completas das partidas:
      - Placares finais (gols marcados/sofridos)
      - Duração das partidas (steps)
      - Vitórias, empates, derrotas
      - Estatísticas agregadas
      - Logging completo no wandb
    """

    def __init__(
        self,
        stage_name: str,
        print_freq_episodes: int = 10,
        verbose: int = 0,
        wandb_run: Optional[Any] = None,
    ):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.print_freq_episodes = print_freq_episodes
        self.wandb_run = wandb_run
        self.episode_count = 0
        
        # Estatísticas por episódio
        self.episode_scores: List[str] = []
        self.match_results: List[Dict[str, Any]] = []
        
        # Estatísticas agregadas (janela deslizante)
        self.window_size = WINDOW_SIZE
        self.recent_matches: collections.deque = collections.deque(maxlen=self.window_size)
        
        # Contadores agregados
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        self.total_goals_scored = 0
        self.total_goals_conceded = 0
        self.total_match_duration = 0

    def _on_step(self) -> bool:
        # infos é uma lista (um por ambiente) em ambientes vetorizados
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        observations = self.locals.get("observations", None)

        for idx, done in enumerate(dones):
            if not done:
                continue
            
            info = infos[idx] if idx < len(infos) else {}
            
            # Tentar obter score do info (adicionado pelo ScoreInfoWrapper quando done=True)
            score = info.get("score")
            
            # Se não estiver no info, tentar obter da observação (fallback)
            if score is None and observations is not None and idx < len(observations):
                obs = observations[idx]
                if isinstance(obs, dict) and "score" in obs:
                    score = obs["score"]
            
            # Se ainda não encontrou, tentar obter do env diretamente (último recurso)
            if score is None:
                # Tentar acessar o ambiente diretamente
                try:
                    envs = self.locals.get("env", None)
                    if envs is not None:
                        # Em ambientes vetorizados, pode ser uma lista ou VecEnv
                        if isinstance(envs, list) and idx < len(envs):
                            env = envs[idx]
                            # Tentar obter do wrapper ScoreInfoWrapper
                            if hasattr(env, "last_observation"):
                                last_obs = env.last_observation
                                if isinstance(last_obs, dict) and "score" in last_obs:
                                    score = last_obs["score"]
                            # Tentar obter do env interno
                            elif hasattr(env, "env"):
                                inner_env = env.env
                                if hasattr(inner_env, "last_observation"):
                                    last_obs = inner_env.last_observation
                                    if isinstance(last_obs, dict) and "score" in last_obs:
                                        score = last_obs["score"]
                except Exception as e:
                    # Silenciosamente ignorar erros de acesso ao env
                    pass
            
            # Se encontrou o score, processar
            # IMPORTANTE: Mesmo se score for None, vamos tentar processar com valores padrão
            # para garantir que as métricas sejam sempre logadas
            if score is not None:
                self.episode_count += 1
                
                # Parse do score (pode ser string, lista, ou array numpy)
                if isinstance(score, str):
                    # Formato "X-Y" ou similar
                    try:
                        parts = score.split("-")
                        if len(parts) >= 2:
                            left_goals, right_goals = map(int, parts[:2])
                        else:
                            left_goals, right_goals = 0, 0
                    except:
                        left_goals, right_goals = 0, 0
                elif isinstance(score, (list, tuple, np.ndarray)):
                    # Score é uma lista/array [left_goals, right_goals]
                    try:
                        if len(score) >= 2:
                            left_goals = int(score[0])
                            right_goals = int(score[1])
                        else:
                            left_goals, right_goals = 0, 0
                    except (ValueError, TypeError, IndexError):
                        left_goals, right_goals = 0, 0
                else:
                    # Tentar converter para lista
                    try:
                        score_list = list(score) if hasattr(score, '__iter__') else [score, 0]
                        if len(score_list) >= 2:
                            left_goals = int(score_list[0])
                            right_goals = int(score_list[1])
                        else:
                            left_goals, right_goals = 0, 0
                    except:
                        left_goals, right_goals = 0, 0
                
                # Determinar resultado (assumindo que o agente joga no time esquerdo)
                if left_goals > right_goals:
                    result = "win"
                    self.total_wins += 1
                elif left_goals < right_goals:
                    result = "loss"
                    self.total_losses += 1
                else:
                    result = "draw"
                    self.total_draws += 1
                
                # Obter duração da partida
                match_duration = info.get("episode", {}).get("l", 0)
                if match_duration == 0:
                    # Tentar obter de steps_left se disponível na observação
                    if observations is not None and idx < len(observations):
                        obs = observations[idx]
                        if isinstance(obs, dict) and "steps_left" in obs:
                            # Duração total - steps restantes (assumindo 3000 steps máx padrão)
                            match_duration = 3000 - obs.get("steps_left", 0)
                
                self.total_goals_scored += left_goals
                self.total_goals_conceded += right_goals
                self.total_match_duration += match_duration
                
                # Calcular diferença de gols
                goal_difference = left_goals - right_goals
                win_by_1plus = (result == "win") and (goal_difference >= MIN_GOAL_DIFFERENCE)
                
                # Criar registro da partida
                match_record = {
                    "episode": self.episode_count,
                    "score": f"{left_goals}-{right_goals}",
                    "left_goals": left_goals,
                    "right_goals": right_goals,
                    "goal_difference": goal_difference,
                    "win_by_1plus_goals": win_by_1plus,
                    "result": result,
                    "duration": match_duration,
                    "timestep": self.num_timesteps,
                }
                
                self.match_results.append(match_record)
                self.recent_matches.append(match_record)
                self.episode_scores.append(f"{left_goals}-{right_goals}")
                
                # Calcular estatísticas da janela recente
                if len(self.recent_matches) > 0:
                    recent_wins = sum(1 for m in self.recent_matches if m["result"] == "win")
                    recent_draws = sum(1 for m in self.recent_matches if m["result"] == "draw")
                    recent_losses = sum(1 for m in self.recent_matches if m["result"] == "loss")
                    recent_goals_scored = sum(m["left_goals"] for m in self.recent_matches)
                    recent_goals_conceded = sum(m["right_goals"] for m in self.recent_matches)
                    recent_avg_duration = np.mean([m["duration"] for m in self.recent_matches])
                    
                    # Métricas de diferença de gols
                    recent_goal_differences = [m["goal_difference"] for m in self.recent_matches]
                    recent_goal_diff_mean = np.mean(recent_goal_differences)
                    recent_goal_diff_std = np.std(recent_goal_differences)
                    recent_wins_by_1plus = sum(1 for m in self.recent_matches if m.get("win_by_1plus_goals", False))
                    win_by_1plus_rate = recent_wins_by_1plus / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Relação de gols (gols marcados / gols sofridos)
                    recent_goals_ratio = (
                        recent_goals_scored / recent_goals_conceded 
                        if recent_goals_conceded > 0 else float('inf')
                    )
                    
                    # Estatísticas de clean sheets (vitórias sem sofrer gols)
                    recent_clean_sheets = sum(1 for m in self.recent_matches 
                                            if m["result"] == "win" and m["right_goals"] == 0)
                    clean_sheet_rate = recent_clean_sheets / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Estatísticas de goleadas (vitórias por 3+ gols de diferença)
                    recent_big_wins = sum(1 for m in self.recent_matches 
                                        if m["result"] == "win" and m["goal_difference"] >= 3)
                    big_win_rate = recent_big_wins / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Estatísticas de derrotas pesadas (derrotas por 3+ gols de diferença)
                    recent_big_losses = sum(1 for m in self.recent_matches 
                                           if m["result"] == "loss" and m["goal_difference"] <= -3)
                    big_loss_rate = recent_big_losses / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Eficiência ofensiva (gols por episódio)
                    recent_offensive_efficiency = recent_goals_scored / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Eficiência defensiva (gols sofridos por episódio)
                    recent_defensive_efficiency = recent_goals_conceded / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Gols por minuto/step (taxa de gols)
                    total_steps = sum(m["duration"] for m in self.recent_matches)
                    goals_per_step = recent_goals_scored / total_steps if total_steps > 0 else 0
                    goals_conceded_per_step = recent_goals_conceded / total_steps if total_steps > 0 else 0
                    
                    # Estatísticas de empates com gols
                    recent_draws_with_goals = sum(1 for m in self.recent_matches 
                                                 if m["result"] == "draw" and (m["left_goals"] > 0 or m["right_goals"] > 0))
                    draw_with_goals_rate = recent_draws_with_goals / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    # Estatísticas de partidas sem gols
                    recent_scoreless = sum(1 for m in self.recent_matches 
                                          if m["left_goals"] == 0 and m["right_goals"] == 0)
                    scoreless_rate = recent_scoreless / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    win_rate = recent_wins / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    draw_rate = recent_draws / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    loss_rate = recent_losses / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                else:
                    win_rate = draw_rate = loss_rate = 0
                    recent_goals_scored = recent_goals_conceded = 0
                    recent_avg_duration = 0
                    recent_goal_diff_mean = recent_goal_diff_std = 0
                    win_by_1plus_rate = 0
                    recent_goals_ratio = 0
                    clean_sheet_rate = 0
                    big_win_rate = 0
                    big_loss_rate = 0
                    recent_offensive_efficiency = 0
                    recent_defensive_efficiency = 0
                    goals_per_step = 0
                    goals_conceded_per_step = 0
                    draw_with_goals_rate = 0
                    scoreless_rate = 0
                
                # Log em wandb, se disponível - SEMPRE logar, mesmo que score seja 0-0
                if self.wandb_run is not None:
                    try:
                        log_dict = {
                            # Estatísticas do episódio atual (SEMPRE logar)
                            "match/score_left": left_goals,
                            "match/score_right": right_goals,
                            "match/goal_difference": goal_difference,
                            "match/result": 1 if result == "win" else (0 if result == "draw" else -1),
                            "match/duration": match_duration,
                            
                            # Estatísticas agregadas (janela recente)
                            "match/win_rate": win_rate,
                            "match/draw_rate": draw_rate,
                            "match/loss_rate": loss_rate,
                            "match/avg_goals_scored": recent_goals_scored / len(self.recent_matches) if len(self.recent_matches) > 0 else 0,
                            "match/avg_goals_conceded": recent_goals_conceded / len(self.recent_matches) if len(self.recent_matches) > 0 else 0,
                            "match/avg_duration": recent_avg_duration,
                            
                            # Métricas de diferença de gols
                            "match/goal_difference_mean": recent_goal_diff_mean,
                            "match/goal_difference_std": recent_goal_diff_std,
                            "match/win_by_1plus_goals_rate": win_by_1plus_rate,
                            
                            # Relações de gols por time
                            "performance/goals_ratio": recent_goals_ratio if recent_goals_ratio != float('inf') else 999.0,
                            "performance/goals_scored_per_episode": recent_offensive_efficiency,
                            "performance/goals_conceded_per_episode": recent_defensive_efficiency,
                            "performance/goals_per_step": goals_per_step,
                            "performance/goals_conceded_per_step": goals_conceded_per_step,
                            
                            # Estatísticas de desempenho defensivo
                            "defense/clean_sheet_rate": clean_sheet_rate,
                            "defense/big_loss_rate": big_loss_rate,
                            "defense/avg_goals_conceded": recent_defensive_efficiency,
                            
                            # Estatísticas de desempenho ofensivo
                            "offense/big_win_rate": big_win_rate,
                            "offense/avg_goals_scored": recent_offensive_efficiency,
                            "offense/goals_per_step": goals_per_step,
                            
                            # Estatísticas de partidas
                            "match/draw_with_goals_rate": draw_with_goals_rate,
                            "match/scoreless_rate": scoreless_rate,
                            
                            # Estatísticas totais
                            "match/total_wins": self.total_wins,
                            "match/total_draws": self.total_draws,
                            "match/total_losses": self.total_losses,
                            "match/total_goals_scored": self.total_goals_scored,
                            "match/total_goals_conceded": self.total_goals_conceded,
                            
                            # Relação total de gols
                            "performance/total_goals_ratio": (
                                self.total_goals_scored / self.total_goals_conceded 
                                if self.total_goals_conceded > 0 else float('inf')
                            ) if (self.total_goals_scored + self.total_goals_conceded) > 0 else 0,
                            
                            # Metadados
                            "episode": self.episode_count,
                            "stage": self.stage_name,
                        }
                        # Tratar infinito para logging
                        if log_dict["performance/goals_ratio"] == float('inf'):
                            log_dict["performance/goals_ratio"] = 999.0
                        if log_dict["performance/total_goals_ratio"] == float('inf'):
                            log_dict["performance/total_goals_ratio"] = 999.0
                            
                        # SEMPRE logar no wandb - garantir que as métricas sejam publicadas
                        # Usar commit=True para forçar o envio imediato
                        self.wandb_run.log(log_dict, step=self.num_timesteps, commit=True)
                    except Exception as e:
                        # Não falhar o treino por causa de logging, mas avisar
                        if self.episode_count % 100 == 0:
                            print(f"⚠ Aviso: Erro ao logar no wandb: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Logar periodicamente mesmo sem episódio completo para garantir visibilidade
                if self.episode_count > 0 and self.episode_count % 50 == 0:
                    try:
                        if self.wandb_run is not None:
                            # Logar estatísticas agregadas mesmo sem novo episódio
                            stats_summary = {
                                "match/total_episodes": self.episode_count,
                                "match/total_wins": self.total_wins,
                                "match/total_draws": self.total_draws,
                                "match/total_losses": self.total_losses,
                                "match/total_goals_scored": self.total_goals_scored,
                                "match/total_goals_conceded": self.total_goals_conceded,
                                "match/overall_win_rate": (
                                    self.total_wins / self.episode_count 
                                    if self.episode_count > 0 else 0
                                ),
                                "match/overall_avg_goals_scored": (
                                    self.total_goals_scored / self.episode_count 
                                    if self.episode_count > 0 else 0
                                ),
                                "match/overall_avg_goals_conceded": (
                                    self.total_goals_conceded / self.episode_count 
                                    if self.episode_count > 0 else 0
                                ),
                            }
                            self.wandb_run.log(stats_summary, step=self.num_timesteps, commit=False)
                    except:
                        pass  # Ignorar erros no log periódico
            else:
                # Se não encontrou score, usar valores padrão (0-0) mas ainda processar
                # para garantir que as métricas sejam sempre logadas
                self.episode_count += 1
                left_goals, right_goals = 0, 0
                result = "draw"
                self.total_draws += 1
                match_duration = info.get("episode", {}).get("l", 0)
                if match_duration == 0 and observations is not None and idx < len(observations):
                    obs = observations[idx]
                    if isinstance(obs, dict) and "steps_left" in obs:
                        match_duration = 3000 - obs.get("steps_left", 0)
                
                self.total_match_duration += match_duration
                goal_difference = 0
                win_by_1plus = False
                
                # Criar registro da partida mesmo sem score
                match_record = {
                    "episode": self.episode_count,
                    "score": "0-0",
                    "left_goals": 0,
                    "right_goals": 0,
                    "goal_difference": 0,
                    "win_by_1plus_goals": False,
                    "result": "draw",
                    "duration": match_duration,
                    "timestep": self.num_timesteps,
                }
                
                self.match_results.append(match_record)
                self.recent_matches.append(match_record)
                self.episode_scores.append("0-0")
                
                # Calcular estatísticas da janela recente
                if len(self.recent_matches) > 0:
                    recent_wins = sum(1 for m in self.recent_matches if m["result"] == "win")
                    recent_draws = sum(1 for m in self.recent_matches if m["result"] == "draw")
                    recent_losses = sum(1 for m in self.recent_matches if m["result"] == "loss")
                    recent_goals_scored = sum(m["left_goals"] for m in self.recent_matches)
                    recent_goals_conceded = sum(m["right_goals"] for m in self.recent_matches)
                    recent_avg_duration = np.mean([m["duration"] for m in self.recent_matches])
                    
                    recent_goal_differences = [m["goal_difference"] for m in self.recent_matches]
                    recent_goal_diff_mean = np.mean(recent_goal_differences)
                    recent_goal_diff_std = np.std(recent_goal_differences)
                    recent_wins_by_1plus = sum(1 for m in self.recent_matches if m.get("win_by_1plus_goals", False))
                    win_by_1plus_rate = recent_wins_by_1plus / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    recent_goals_ratio = (
                        recent_goals_scored / recent_goals_conceded 
                        if recent_goals_conceded > 0 else float('inf')
                    )
                    
                    recent_clean_sheets = sum(1 for m in self.recent_matches 
                                            if m["result"] == "win" and m["right_goals"] == 0)
                    clean_sheet_rate = recent_clean_sheets / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    recent_big_wins = sum(1 for m in self.recent_matches 
                                        if m["result"] == "win" and m["goal_difference"] >= 3)
                    big_win_rate = recent_big_wins / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    recent_big_losses = sum(1 for m in self.recent_matches 
                                           if m["result"] == "loss" and m["goal_difference"] <= -3)
                    big_loss_rate = recent_big_losses / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    recent_offensive_efficiency = recent_goals_scored / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    recent_defensive_efficiency = recent_goals_conceded / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    total_steps = sum(m["duration"] for m in self.recent_matches)
                    goals_per_step = recent_goals_scored / total_steps if total_steps > 0 else 0
                    goals_conceded_per_step = recent_goals_conceded / total_steps if total_steps > 0 else 0
                    
                    recent_draws_with_goals = sum(1 for m in self.recent_matches 
                                                 if m["result"] == "draw" and (m["left_goals"] > 0 or m["right_goals"] > 0))
                    draw_with_goals_rate = recent_draws_with_goals / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    recent_scoreless = sum(1 for m in self.recent_matches 
                                          if m["left_goals"] == 0 and m["right_goals"] == 0)
                    scoreless_rate = recent_scoreless / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    
                    win_rate = recent_wins / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    draw_rate = recent_draws / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    loss_rate = recent_losses / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                else:
                    win_rate = draw_rate = loss_rate = 0
                    recent_goals_scored = recent_goals_conceded = 0
                    recent_avg_duration = 0
                    recent_goal_diff_mean = recent_goal_diff_std = 0
                    win_by_1plus_rate = 0
                    recent_goals_ratio = 0
                    clean_sheet_rate = 0
                    big_win_rate = 0
                    big_loss_rate = 0
                    recent_offensive_efficiency = 0
                    recent_defensive_efficiency = 0
                    goals_per_step = 0
                    goals_conceded_per_step = 0
                    draw_with_goals_rate = 0
                    scoreless_rate = 0
                
                # Log em wandb mesmo sem score encontrado (usando valores padrão)
                if self.wandb_run is not None:
                    try:
                        log_dict = {
                            "match/score_left": 0,
                            "match/score_right": 0,
                            "match/goal_difference": 0,
                            "match/result": 0,
                            "match/duration": match_duration,
                            "match/win_rate": win_rate,
                            "match/draw_rate": draw_rate,
                            "match/loss_rate": loss_rate,
                            "match/avg_goals_scored": recent_goals_scored / len(self.recent_matches) if len(self.recent_matches) > 0 else 0,
                            "match/avg_goals_conceded": recent_goals_conceded / len(self.recent_matches) if len(self.recent_matches) > 0 else 0,
                            "match/avg_duration": recent_avg_duration,
                            "match/goal_difference_mean": recent_goal_diff_mean,
                            "match/goal_difference_std": recent_goal_diff_std,
                            "match/win_by_1plus_goals_rate": win_by_1plus_rate,
                            "performance/goals_ratio": recent_goals_ratio if recent_goals_ratio != float('inf') else 999.0,
                            "performance/goals_scored_per_episode": recent_offensive_efficiency,
                            "performance/goals_conceded_per_episode": recent_defensive_efficiency,
                            "performance/goals_per_step": goals_per_step,
                            "performance/goals_conceded_per_step": goals_conceded_per_step,
                            "defense/clean_sheet_rate": clean_sheet_rate,
                            "defense/big_loss_rate": big_loss_rate,
                            "defense/avg_goals_conceded": recent_defensive_efficiency,
                            "offense/big_win_rate": big_win_rate,
                            "offense/avg_goals_scored": recent_offensive_efficiency,
                            "offense/goals_per_step": goals_per_step,
                            "match/draw_with_goals_rate": draw_with_goals_rate,
                            "match/scoreless_rate": scoreless_rate,
                            "match/total_wins": self.total_wins,
                            "match/total_draws": self.total_draws,
                            "match/total_losses": self.total_losses,
                            "match/total_goals_scored": self.total_goals_scored,
                            "match/total_goals_conceded": self.total_goals_conceded,
                            "performance/total_goals_ratio": (
                                self.total_goals_scored / self.total_goals_conceded 
                                if self.total_goals_conceded > 0 else float('inf')
                            ) if (self.total_goals_scored + self.total_goals_conceded) > 0 else 0,
                            "episode": self.episode_count,
                            "stage": self.stage_name,
                            "match/score_not_found": 1,  # Flag para indicar que score não foi encontrado
                        }
                        if log_dict["performance/goals_ratio"] == float('inf'):
                            log_dict["performance/goals_ratio"] = 999.0
                        if log_dict["performance/total_goals_ratio"] == float('inf'):
                            log_dict["performance/total_goals_ratio"] = 999.0
                        
                        self.wandb_run.log(log_dict, step=self.num_timesteps, commit=True)
                    except Exception as e:
                        if self.episode_count % 100 == 0:
                            print(f"⚠ Aviso: Erro ao logar no wandb (sem score): {e}")
                
                # Logar aviso ocasionalmente
                if self.episode_count % 100 == 0:
                    print(f"⚠ Aviso: Score não encontrado para episódio {self.episode_count} (usando 0-0)")
                    print(f"  Info keys: {list(info.keys()) if info else 'None'}")
                    if observations is not None and idx < len(observations):
                        obs = observations[idx]
                        if isinstance(obs, dict):
                            print(f"  Observation keys: {list(obs.keys())[:10]}...")

                # Impressão periódica
                if (
                    self.print_freq_episodes > 0
                    and self.episode_count % self.print_freq_episodes == 0
                ):
                    last_scores = ", ".join(
                        self.episode_scores[-self.print_freq_episodes :]
                    )
                    print(
                        f"[{self.stage_name}] Episódios "
                        f"{self.episode_count-self.print_freq_episodes+1}"
                        f"-{self.episode_count} placares: {last_scores}"
                    )
                    print(
                        f"  Taxa de vitória (últimos {len(self.recent_matches)}): "
                        f"{win_rate:.1%} | Empates: {draw_rate:.1%} | Derrotas: {loss_rate:.1%}"
                    )
                    print(
                        f"  Gols médios: {recent_goals_scored/len(self.recent_matches):.2f} marcados, "
                        f"{recent_goals_conceded/len(self.recent_matches):.2f} sofridos"
                    )

        return True
    
    def get_match_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas agregadas das partidas."""
        total_matches = self.total_wins + self.total_draws + self.total_losses
        if total_matches == 0:
            return {}
        
        # Calcular métricas de diferença de gols
        goal_differences = [m.get("goal_difference", 0) for m in self.match_results]
        wins_by_1plus = sum(1 for m in self.match_results if m.get("win_by_1plus_goals", False))
        
        # Estatísticas adicionais de desempenho
        clean_sheets = sum(1 for m in self.match_results 
                          if m["result"] == "win" and m["right_goals"] == 0)
        big_wins = sum(1 for m in self.match_results 
                      if m["result"] == "win" and m["goal_difference"] >= 3)
        big_losses = sum(1 for m in self.match_results 
                        if m["result"] == "loss" and m["goal_difference"] <= -3)
        scoreless_matches = sum(1 for m in self.match_results 
                               if m["left_goals"] == 0 and m["right_goals"] == 0)
        
        # Relação de gols
        goals_ratio = (
            self.total_goals_scored / self.total_goals_conceded 
            if self.total_goals_conceded > 0 else float('inf')
        )
        
        # Gols por step
        goals_per_step = (
            self.total_goals_scored / self.total_match_duration 
            if self.total_match_duration > 0 else 0
        )
        goals_conceded_per_step = (
            self.total_goals_conceded / self.total_match_duration 
            if self.total_match_duration > 0 else 0
        )
        
        return {
            "total_episodes": self.episode_count,
            "total_wins": self.total_wins,
            "total_draws": self.total_draws,
            "total_losses": self.total_losses,
            "win_rate": self.total_wins / total_matches,
            "draw_rate": self.total_draws / total_matches,
            "loss_rate": self.total_losses / total_matches,
            "avg_goals_scored": self.total_goals_scored / total_matches,
            "avg_goals_conceded": self.total_goals_conceded / total_matches,
            "avg_match_duration": self.total_match_duration / total_matches,
            "goal_difference_mean": np.mean(goal_differences) if goal_differences else 0,
            "goal_difference_std": np.std(goal_differences) if goal_differences else 0,
            "win_by_1plus_goals_rate": wins_by_1plus / total_matches if total_matches > 0 else 0,
            # Relações de gols
            "goals_ratio": goals_ratio if goals_ratio != float('inf') else 999.0,
            "goals_per_step": goals_per_step,
            "goals_conceded_per_step": goals_conceded_per_step,
            # Estatísticas de desempenho
            "clean_sheet_rate": clean_sheets / total_matches if total_matches > 0 else 0,
            "big_win_rate": big_wins / total_matches if total_matches > 0 else 0,
            "big_loss_rate": big_losses / total_matches if total_matches > 0 else 0,
            "scoreless_rate": scoreless_matches / total_matches if total_matches > 0 else 0,
        }


class AdaptiveCurriculum:
    """
    Sistema de curriculum learning adaptativo baseado em performance.
    Monitora métricas e decide quando avançar ou regredir de estágio.
    """
    
    def __init__(
        self,
        stages: List[Dict[str, Any]],
        min_win_rate: float = MIN_WIN_RATE_TO_ADVANCE,
        min_episodes: int = MIN_EPISODES_FOR_EVAL,
        window_size: int = WINDOW_SIZE,
    ):
        self.stages = stages
        self.min_win_rate = min_win_rate
        self.min_episodes = min_episodes
        self.window_size = window_size
        
        self.current_stage_idx = 0
        self.stage_metrics: Dict[str, List[Dict[str, Any]]] = {
            stage["name"]: [] for stage in stages
        }
        
        # Sistema de estabilidade: rastrear avaliações consecutivas que atendem critérios
        self.stability_tracker: Dict[str, int] = {
            stage["name"]: 0 for stage in stages
        }
        
        # Sistema de regressão: rastrear avaliações consecutivas com performance baixa
        self.regression_tracker: Dict[str, int] = {
            stage["name"]: 0 for stage in stages
        }
        self.last_regression_chunk: Dict[str, int] = {
            stage["name"]: -1 for stage in stages
        }
        
    def update_metrics(self, stage_name: str, episode_stats: Dict[str, Any]):
        """Atualiza métricas de um episódio para o estágio atual."""
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = []
        
        self.stage_metrics[stage_name].append(episode_stats)
        
        # Manter apenas a janela mais recente
        if len(self.stage_metrics[stage_name]) > self.window_size:
            self.stage_metrics[stage_name] = self.stage_metrics[stage_name][-self.window_size:]
    
    def should_advance(self, stage_name: str) -> bool:
        """
        Decide se deve avançar para o próximo estágio.
        Requer critérios rigorosos:
        - Taxa de vitória >= 70%
        - 90% das vitórias com >= 1 gol de diferença
        - Gols médios marcados >= 1.0
        - Gols médios sofridos < 0.5
        - Estabilidade: critérios mantidos por STABILITY_EVALUATIONS avaliações consecutivas
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Já está no último estágio
        
        metrics = self.stage_metrics.get(stage_name, [])
        if len(metrics) < self.min_episodes:
            return False  # Não tem episódios suficientes
        
        # Calcular métricas na janela recente
        recent_metrics = metrics[-self.min_episodes:]
        
        # 1. Taxa de vitória
        wins = sum(1 for m in recent_metrics if m.get("result") == "win")
        win_rate = wins / len(recent_metrics)
        
        # 2. Gols médios marcados e sofridos
        avg_goals_scored = np.mean([m.get("left_goals", 0) for m in recent_metrics])
        avg_goals_conceded = np.mean([m.get("right_goals", 0) for m in recent_metrics])
        
        # 3. Taxa de vitórias com >= 1 gol de diferença
        wins_list = [m for m in recent_metrics if m.get("result") == "win"]
        if len(wins_list) > 0:
            wins_by_1plus = sum(1 for m in wins_list if m.get("win_by_1plus_goals", False))
            win_by_1plus_rate = wins_by_1plus / len(wins_list)
        else:
            win_by_1plus_rate = 0.0
        
        # Verificar todos os critérios
        criteria_met = (
            win_rate >= self.min_win_rate and
            win_by_1plus_rate >= MIN_WIN_BY_GOAL_DIFFERENCE_RATE and
            avg_goals_scored >= MIN_GOALS_SCORED and
            avg_goals_conceded < MAX_GOALS_CONCEDED
        )
        
        # Atualizar tracker de estabilidade
        if criteria_met:
            self.stability_tracker[stage_name] += 1
        else:
            self.stability_tracker[stage_name] = 0
        
        # Avançar apenas se critérios foram mantidos por STABILITY_EVALUATIONS avaliações consecutivas
        should_advance = (
            criteria_met and 
            self.stability_tracker[stage_name] >= STABILITY_EVALUATIONS
        )
        
        if should_advance:
            print(
                f"✓ Critério de progressão atingido para '{stage_name}' "
                f"(estabilidade: {self.stability_tracker[stage_name]}/{STABILITY_EVALUATIONS}):"
            )
            print(f"  - Taxa de vitória: {win_rate:.1%} (>= {self.min_win_rate:.1%})")
            print(f"  - Vitórias com >=1 gol: {win_by_1plus_rate:.1%} (>= {MIN_WIN_BY_GOAL_DIFFERENCE_RATE:.1%})")
            print(f"  - Gols médios marcados: {avg_goals_scored:.2f} (>= {MIN_GOALS_SCORED:.1f})")
            print(f"  - Gols médios sofridos: {avg_goals_conceded:.2f} (< {MAX_GOALS_CONCEDED:.1f})")
        elif criteria_met:
            print(
                f"  Progresso em '{stage_name}': Critérios atendidos "
                f"({self.stability_tracker[stage_name]}/{STABILITY_EVALUATIONS} avaliações consecutivas)"
            )
        
        return should_advance
    
    def should_regress(self, stage_name: str, current_chunk: int) -> bool:
        """
        Decide se deve regredir para o estágio anterior.
        Critérios para regressão:
        - Taxa de vitória < REGRESSION_WIN_RATE_THRESHOLD por REGRESSION_EVALUATIONS avaliações consecutivas
        - Diferença média de gols negativa por REGRESSION_EVALUATIONS avaliações consecutivas
        - Cooldown para evitar oscilação entre estágios
        """
        if self.current_stage_idx == 0:
            return False  # Já está no primeiro estágio
        
        # Verificar cooldown
        if self.last_regression_chunk[stage_name] >= 0:
            chunks_since_regression = current_chunk - self.last_regression_chunk[stage_name]
            if chunks_since_regression < REGRESSION_COOLDOWN:
                return False  # Ainda em cooldown
        
        metrics = self.stage_metrics.get(stage_name, [])
        if len(metrics) < self.min_episodes:
            return False  # Não tem dados suficientes
        
        # Calcular métricas na janela recente
        recent_metrics = metrics[-self.min_episodes:]
        wins = sum(1 for m in recent_metrics if m.get("result") == "win")
        win_rate = wins / len(recent_metrics)
        
        # Calcular diferença média de gols
        avg_goal_diff = np.mean([m.get("goal_difference", 0) for m in recent_metrics])
        
        # Verificar critérios de regressão
        should_regress_this_eval = (
            win_rate < REGRESSION_WIN_RATE_THRESHOLD or
            avg_goal_diff < 0
        )
        
        # Atualizar tracker de regressão
        if should_regress_this_eval:
            self.regression_tracker[stage_name] += 1
        else:
            self.regression_tracker[stage_name] = 0
        
        # Regredir apenas se critérios foram mantidos por REGRESSION_EVALUATIONS avaliações consecutivas
        should_regress = (
            should_regress_this_eval and
            self.regression_tracker[stage_name] >= REGRESSION_EVALUATIONS
        )
        
        if should_regress:
            print(
                f"⚠ Performance muito baixa em '{stage_name}' "
                f"(regressão: {self.regression_tracker[stage_name]}/{REGRESSION_EVALUATIONS}):"
            )
            print(f"  - Taxa de vitória: {win_rate:.1%} (< {REGRESSION_WIN_RATE_THRESHOLD:.1%})")
            print(f"  - Diferença média de gols: {avg_goal_diff:.2f} (< 0)")
            print(f"  - Regredindo para estágio anterior...")
            self.last_regression_chunk[stage_name] = current_chunk
        
        return should_regress
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Retorna o estágio atual."""
        return self.stages[self.current_stage_idx]
    
    def advance(self):
        """Avança para o próximo estágio."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            print(f"→ Avançando para estágio: {self.stages[self.current_stage_idx]['name']}")
    
    def regress(self):
        """Regride para o estágio anterior."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            print(f"← Regredindo para estágio: {self.stages[self.current_stage_idx]['name']}")
    
    def get_current_difficulty(self) -> Dict[str, Any]:
        """Retorna configuração atual de dificuldade."""
        stage = self.get_current_stage()
        metrics = self.stage_metrics.get(stage["name"], [])
        
        if len(metrics) == 0:
            return {"stage": stage, "win_rate": 0.0, "episodes": 0}
        
        recent_metrics = metrics[-self.min_episodes:] if len(metrics) >= self.min_episodes else metrics
        wins = sum(1 for m in recent_metrics if m.get("result") == "win")
        win_rate = wins / len(recent_metrics) if len(recent_metrics) > 0 else 0
        
        return {
            "stage": stage,
            "win_rate": win_rate,
            "episodes": len(metrics),
            "recent_episodes": len(recent_metrics),
        }


class SelfPlayManager:
    """
    Gerencia pool de checkpoints para self-play.
    Mantém versões anteriores do modelo e seleciona oponentes.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = SELF_PLAY_POOL_SIZE,
        strategy: str = "balanced",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.strategy = strategy  # "balanced", "random", "stronger", "weaker"
        
        # Pool de checkpoints: [(path, performance_metrics, timestep), ...]
        self.checkpoint_pool: List[Tuple[str, Dict[str, Any], int]] = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def add_checkpoint(self, checkpoint_path: str, performance_metrics: Dict[str, Any], timestep: int):
        """Adiciona um checkpoint ao pool."""
        if not os.path.exists(checkpoint_path):
            print(f"⚠ Aviso: Checkpoint não encontrado: {checkpoint_path}")
            return
        
        # Adicionar ao pool
        self.checkpoint_pool.append((checkpoint_path, performance_metrics, timestep))
        
        # Ordenar por timestep (mais recente primeiro)
        self.checkpoint_pool.sort(key=lambda x: x[2], reverse=True)
        
        # Limpar checkpoints antigos
        self.cleanup_old_checkpoints()
        
        print(f"✓ Checkpoint adicionado ao pool de self-play: {checkpoint_path}")
    
    def select_opponent(self) -> Optional[str]:
        """
        Seleciona um checkpoint oponente baseado na estratégia.
        Retorna o caminho do checkpoint ou None se não houver oponentes.
        """
        if len(self.checkpoint_pool) == 0:
            return None
        
        if self.strategy == "random":
            import random
            return random.choice(self.checkpoint_pool)[0]
        
        elif self.strategy == "balanced":
            # Seleciona aleatoriamente entre os últimos N checkpoints (mais balanceado)
            import random
            n_recent = min(5, len(self.checkpoint_pool))
            return random.choice(self.checkpoint_pool[:n_recent])[0]
        
        elif self.strategy == "stronger":
            # Seleciona o checkpoint com melhor performance
            best_checkpoint = max(self.checkpoint_pool, key=lambda x: x[1].get("win_rate", 0))
            return best_checkpoint[0]
        
        elif self.strategy == "weaker":
            # Seleciona o checkpoint com pior performance (para treinar contra oponentes mais fracos)
            worst_checkpoint = min(self.checkpoint_pool, key=lambda x: x[1].get("win_rate", 1.0))
            return worst_checkpoint[0]
        
        else:
            # Padrão: mais recente
            return self.checkpoint_pool[0][0]
    
    def should_use_self_play(self, stage_name: str) -> bool:
        """Decide se deve usar self-play neste batch (sempre ativo)."""
        # Verificar se o estágio está na lista de estágios com self-play
        if SELF_PLAY_STAGES and stage_name not in SELF_PLAY_STAGES:
            return False
        
        # Verificar se há checkpoints disponíveis
        if len(self.checkpoint_pool) == 0:
            return False
        
        # Usar self-play com probabilidade SELF_PLAY_RATIO
        import random
        return random.random() < SELF_PLAY_RATIO
    
    def cleanup_old_checkpoints(self):
        """Remove checkpoints antigos do pool."""
        if len(self.checkpoint_pool) <= self.max_checkpoints:
            return
        
        # Manter apenas os N mais recentes
        self.checkpoint_pool = self.checkpoint_pool[:self.max_checkpoints]
    
    def get_pool_size(self) -> int:
        """Retorna o tamanho atual do pool."""
        return len(self.checkpoint_pool)


def get_stage_hyperparams(stage_name: str) -> Dict[str, Any]:
    """
    Retorna hiperparâmetros específicos para cada estágio.
    Estágios iniciais: learning rate mais alto, mais exploração
    Estágios avançados (11v11): learning rate mais baixo, menos exploração
    """
    if "stage1" in stage_name or "stage2" in stage_name:
        # Estágios iniciais: mais exploração e learning rate mais alto
        return {
            "learning_rate": LR * 1.2,  # 20% mais alto
            "ent_coef": ENT_COEF * 1.5,  # Mais exploração
            "clip_range": CLIP_RANGE,
            "n_epochs": N_EPOCHS,
        }
    elif "stage3" in stage_name or "11v11" in stage_name:
        # Estágio 11v11: learning rate mais baixo, menos exploração
        return {
            "learning_rate": LR * 0.7,  # 30% mais baixo
            "ent_coef": ENT_COEF * 0.7,  # Menos exploração
            "clip_range": CLIP_RANGE * 0.9,  # Clipping mais conservador
            "n_epochs": N_EPOCHS + 1,  # Mais épocas para estabilidade
        }
    else:
        # Estágios intermediários: valores padrão
        return {
            "learning_rate": LR,
            "ent_coef": ENT_COEF,
            "clip_range": CLIP_RANGE,
            "n_epochs": N_EPOCHS,
        }


def make_env_with_opponent(level: str, log_dir: str, opponent_model_path: Optional[str] = None):
    """
    Cria ambiente do gfootball, opcionalmente com oponente controlado por modelo.
    Se opponent_model_path for None, usa bots padrão.
    
    Nota: A integração completa de self-play requereria um wrapper mais complexo
    que carrega o modelo oponente e intercepta ações do time direito.
    Por enquanto, retorna o ambiente padrão (self-play será implementado futuramente).
    """
    try:
        # Configurar tipo de recompensas
        rewards_config = "scoring"
        if USE_CHECKPOINT_REWARDS:
            rewards_config = "scoring,checkpoints"
        
        env = football_env.create_environment(
            env_name=level,
            stacked=True,
            representation="extracted",
            rewards=rewards_config,
            logdir=log_dir,
            render=False,
        )
        
        # Envolver com wrapper de segurança primeiro
        env = SafeWrapper(env, max_retries=3)
        # Depois adicionar wrapper que adiciona score ao info
        env = ScoreInfoWrapper(env)
        # Adicionar wrapper de recompensas densas se habilitado
        if USE_DENSE_REWARDS:
            env = DenseRewardWrapper(env, reward_scale=DENSE_REWARD_SCALE)
        
        # TODO: Implementar carregamento e integração do modelo oponente
        # Isso requereria:
        # 1. Carregar o modelo do checkpoint
        # 2. Criar um wrapper que intercepta ações do time direito
        # 3. Usar o modelo para gerar ações do oponente
        
        return env
    except Exception as e:
        print(f"⚠ Erro ao criar ambiente com oponente {level}: {e}")
        traceback.print_exc()
        raise


def validate_curriculum_setup() -> bool:
    """
    Valida se todos os estágios do curriculum podem ser executados.
    Verifica:
    - Existência e criação de cada nível/cenário
    - Criação de diretórios necessários
    - Recursos do sistema (GPU, memória)
    - Configurações básicas
    
    Retorna True se tudo estiver OK, False caso contrário.
    """
    # Importar football_env explicitamente para evitar problemas de escopo
    import gfootball.env as football_env
    
    print("\n" + "="*80)
    print("VALIDANDO CONFIGURAÇÃO DO CURRICULUM LEARNING")
    print("="*80)
    
    all_checks_passed = True
    
    # ==========
    # Check 1: Verificar GPU e recursos
    # ==========
    print("\n[CHECK 1/5] Verificando recursos do sistema...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")
            if gpu_memory < 20:
                print(f"  ⚠ Aviso: GPU com menos de 20GB pode ter problemas com {NUM_ENVS} ambientes")
            else:
                print(f"  ✓ Memória GPU suficiente para {NUM_ENVS} ambientes")
        else:
            print(f"  ⚠ GPU não disponível, usando CPU (pode ser muito lento)")
    except ImportError:
        print(f"  ⚠ PyTorch não disponível")
    
    print(f"  ✓ Número de ambientes: {NUM_ENVS}")
    print(f"  ✓ Batch size calculado: {(N_STEPS * NUM_ENVS) // N_MINIBATCHES}")
    
    # ==========
    # Check 2: Verificar diretórios
    # ==========
    print("\n[CHECK 2/5] Verificando diretórios...")
    directories = {
        "Logs": LOG_DIR_BASE,
        "Checkpoints": CHECKPOINT_DIR_BASE,
        "Avaliações": EVAL_DIR_BASE,
    }
    
    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.exists(path) and os.path.isdir(path):
                # Verificar permissões de escrita
                test_file = os.path.join(path, ".test_write")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    print(f"  ✓ {name}: {path} (criado e com permissão de escrita)")
                except Exception as e:
                    print(f"  ✗ {name}: {path} (sem permissão de escrita: {e})")
                    all_checks_passed = False
            else:
                print(f"  ✗ {name}: {path} (não foi possível criar)")
                all_checks_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {path} (erro: {e})")
            all_checks_passed = False
    
    # ==========
    # Check 3: Verificar cada estágio do curriculum
    # ==========
    print("\n[CHECK 3/5] Verificando estágios do curriculum...")
    print(f"  Total de estágios: {len(CURRICULUM_STAGES)}")
    
    for idx, stage in enumerate(CURRICULUM_STAGES):
        stage_name = stage["name"]
        level = stage["level"]
        total_timesteps = stage["total_timesteps"]
        
        print(f"\n  Estágio {idx+1}/{len(CURRICULUM_STAGES)}: {stage_name}")
        print(f"    Nível: {level}")
        print(f"    Timesteps planejados: {total_timesteps:,}")
        
        # Verificar se o nível pode ser criado
        try:
            test_log_dir = os.path.join(LOG_DIR_BASE, f".test_{stage_name}")
            os.makedirs(test_log_dir, exist_ok=True)
            
            # Tentar criar o ambiente
            print(f"    → Testando criação do ambiente...", end=" ", flush=True)
            test_env = football_env.create_environment(
                env_name=level,
                stacked=True,
                representation="extracted",
                rewards="scoring",
                logdir=test_log_dir,
                render=False,
            )
            
            # Testar reset
            obs = test_env.reset()
            if obs is not None:
                print("✓")
                print(f"    ✓ Ambiente criado com sucesso")
                print(f"    ✓ Observação shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            else:
                print("✗")
                print(f"    ✗ Ambiente criado mas reset() retornou None")
                all_checks_passed = False
            
            # Fechar ambiente
            test_env.close()
            
            # Limpar diretório de teste
            try:
                if os.path.exists(test_log_dir):
                    shutil.rmtree(test_log_dir)
            except:
                pass
                
        except Exception as e:
            print("✗")
            print(f"    ✗ Erro ao criar ambiente: {e}")
            all_checks_passed = False
            import traceback
            traceback.print_exc()
        
        # Verificar diretórios do estágio
        stage_log_dir = os.path.join(LOG_DIR_BASE, stage_name)
        stage_checkpoint_dir = os.path.join(CHECKPOINT_DIR_BASE, stage_name)
        stage_eval_dir = os.path.join(EVAL_DIR_BASE, stage_name)
        
        try:
            os.makedirs(stage_log_dir, exist_ok=True)
            os.makedirs(stage_checkpoint_dir, exist_ok=True)
            os.makedirs(stage_eval_dir, exist_ok=True)
            print(f"    ✓ Diretórios do estágio criados")
        except Exception as e:
            print(f"    ✗ Erro ao criar diretórios: {e}")
            all_checks_passed = False
    
    # ==========
    # Check 4: Verificar configurações do curriculum adaptativo
    # ==========
    print("\n[CHECK 4/5] Verificando configurações do curriculum adaptativo...")
    print(f"  ✓ Curriculum adaptativo: {'Ativado' if ADAPTIVE_CURRICULUM else 'Desativado'}")
    print(f"  ✓ Taxa mínima de vitória para avançar: {MIN_WIN_RATE_TO_ADVANCE:.1%}")
    print(f"  ✓ Episódios mínimos para avaliação: {MIN_EPISODES_FOR_EVAL}")
    print(f"  ✓ Tamanho da janela: {WINDOW_SIZE}")
    print(f"  ✓ Diferença mínima de gols: {MIN_GOAL_DIFFERENCE}")
    print(f"  ✓ Taxa mínima de vitórias com >=1 gol: {MIN_WIN_BY_GOAL_DIFFERENCE_RATE:.1%}")
    print(f"  ✓ Gols médios mínimos marcados: {MIN_GOALS_SCORED:.1f}")
    print(f"  ✓ Gols médios máximos sofridos: {MAX_GOALS_CONCEDED:.1f}")
    print(f"  ✓ Avaliações de estabilidade necessárias: {STABILITY_EVALUATIONS}")
    print(f"  ✓ Timesteps máximos por estágio: {MAX_TIMESTEPS_PER_STAGE:,}")
    print(f"  ✓ Tamanho do chunk de avaliação: {EVAL_CHUNK_SIZE:,}")
    
    # ==========
    # Check 5: Verificar self-play e dependências
    # ==========
    print("\n[CHECK 5/5] Verificando self-play e dependências...")
    print(f"  ✓ Self-play: {'Ativado' if ENABLE_SELF_PLAY else 'Desativado'}")
    if ENABLE_SELF_PLAY:
        print(f"  ✓ Taxa de self-play: {SELF_PLAY_RATIO:.1%}")
        print(f"  ✓ Tamanho do pool: {SELF_PLAY_POOL_SIZE}")
    
    # Verificar dependências
    dependencies = {
        "stable_baselines3": "PPO",
        "gfootball": "create_environment",
        "numpy": "np",
        "torch": "PyTorch",
    }
    
    for module_name, description in dependencies.items():
        try:
            if module_name == "stable_baselines3":
                from stable_baselines3 import PPO
            elif module_name == "gfootball":
                import gfootball.env as football_env
            elif module_name == "numpy":
                import numpy
            elif module_name == "torch":
                import torch
            print(f"  ✓ {description} ({module_name}) disponível")
        except ImportError as e:
            print(f"  ✗ {description} ({module_name}) não disponível: {e}")
            all_checks_passed = False
    
    # Verificar wandb
    try:
        import wandb
        print(f"  ✓ wandb disponível")
    except ImportError:
        print(f"  ⚠ wandb não disponível (logging será desabilitado)")
    
    # ==========
    # Resumo final
    # ==========
    print("\n" + "="*80)
    if all_checks_passed:
        print("✓ TODAS AS VALIDAÇÕES PASSARAM - Pronto para iniciar treinamento!")
        print("="*80)
    else:
        print("✗ ALGUMAS VALIDAÇÕES FALHARAM - Corrija os problemas antes de continuar")
        print("="*80)
    
    return all_checks_passed


def signal_handler(sig, frame):
    """Handler para capturar sinais e salvar estado antes de sair."""
    print("\n⚠ Sinal de interrupção recebido (SIGINT/SIGTERM). Salvando estado...")
    # O modelo será salvo no tratamento de exceção
    sys.exit(0)


def main():
    # Habilitar faulthandler para melhor captura de SIGSEGV e outros erros
    try:
        import faulthandler
        # Habilitar faulthandler para capturar stack traces em caso de crash
        faulthandler.enable()
        # Registrar dump de SIGSEGV em arquivo (LOG_DIR_BASE já está definido antes de main)
        try:
            os.makedirs(LOG_DIR_BASE, exist_ok=True)
            segfault_log = os.path.join(LOG_DIR_BASE, "segfault_dump.log")
            segfault_file = open(segfault_log, 'w')
            faulthandler.register(signal.SIGSEGV, file=segfault_file, all_threads=True)
            print(f"✓ Faulthandler habilitado - dumps de SIGSEGV serão salvos em: {segfault_log}")
        except Exception as e:
            print(f"⚠ Não foi possível registrar handler de SIGSEGV: {e}")
            print("  (continuando sem registro de SIGSEGV)")
            # Ainda habilitar faulthandler básico
            faulthandler.enable()
    except ImportError:
        print("⚠ faulthandler não disponível (Python < 3.3)")
    except Exception as e:
        print(f"⚠ Erro ao configurar faulthandler: {e}")
    
    # Registrar handlers de sinais para capturar interrupções
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ==========
    # Validação pré-treinamento
    # ==========
    print("\n" + "="*80)
    print("INICIANDO VALIDAÇÃO DO CURRICULUM LEARNING")
    print("="*80)
    
    if not validate_curriculum_setup():
        print("\n❌ VALIDAÇÃO FALHOU - Abortando treinamento")
        print("Por favor, corrija os problemas acima antes de continuar.")
        return
    
    print("\n" + "="*80)
    print("INICIANDO TREINAMENTO")
    print("="*80 + "\n")
    
    # ==========
    # wandb (sempre ativado)
    # ==========
    import wandb

    # Verificar se wandb está instalado e funcionando
    print("Inicializando wandb...")
    
    wandb_project = os.environ.get("WANDB_PROJECT", "gfootball_repro_scoring")
    wandb_run_name_base = os.environ.get(
        "WANDB_RUN_NAME",
        f"curriculum_gfootball_ne{NUM_ENVS}",
    )
    
    # Config básica enviada para o wandb
    wandb_config = {
        "num_envs": NUM_ENVS,
        "n_steps": N_STEPS,
        "n_epochs": N_EPOCHS,
        "n_minibatches": N_MINIBATCHES,
        "lr": LR,
        "gamma": GAMMA,
        "ent_coef": ENT_COEF,
        "clip_range": CLIP_RANGE,
        "gae_lambda": GAE_LAMBDA,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "curriculum_stages": CURRICULUM_STAGES,
        "adaptive_curriculum": ADAPTIVE_CURRICULUM,
        "enable_self_play": ENABLE_SELF_PLAY,
        "self_play_ratio": SELF_PLAY_RATIO,
        "min_win_rate_to_advance": MIN_WIN_RATE_TO_ADVANCE,
        "checkpoint_freq": CHECKPOINT_FREQ,
        # Novos critérios rigorosos
        "min_goal_difference": MIN_GOAL_DIFFERENCE,
        "min_win_by_goal_difference_rate": MIN_WIN_BY_GOAL_DIFFERENCE_RATE,
        "min_goals_scored": MIN_GOALS_SCORED,
        "max_goals_conceded": MAX_GOALS_CONCEDED,
        "stability_evaluations": STABILITY_EVALUATIONS,
        "max_timesteps_per_stage": MAX_TIMESTEPS_PER_STAGE,
        "eval_chunk_size": EVAL_CHUNK_SIZE,
        "regression_win_rate_threshold": REGRESSION_WIN_RATE_THRESHOLD,
        "regression_evaluations": REGRESSION_EVALUATIONS,
        "regression_cooldown": REGRESSION_COOLDOWN,
    }
    
    # Forçar modo interativo para pedir credenciais se necessário
    # Se WANDB_MODE não estiver definido, usar "online" (padrão)
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    
    # Criar run principal do wandb (para tracking geral do curriculum)
    wandb_run_main = wandb.init(
        project=wandb_project,
        name=wandb_run_name_base,
        config=wandb_config,
        sync_tensorboard=True,
        mode=wandb_mode,
        tags=["curriculum_main"],
    )
    
    # Verificar se a inicialização foi bem-sucedida
    if wandb_run_main is not None:
        print(f"✓ wandb inicializado com sucesso!")
        print(f"  Projeto: {wandb_project}")
        print(f"  Run principal: {wandb_run_name_base}")
        print(f"  URL: {wandb_run_main.url if hasattr(wandb_run_main, 'url') else 'N/A'}")
    else:
        print("⚠ Aviso: wandb.init() retornou None")
    
    # Variável para armazenar o run atual do estágio
    wandb_run_stage = None

    # ==========
    # Detecção de GPU e configuração de dispositivo
    # ==========
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU detectada: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠ GPU não disponível, usando CPU")
    except ImportError:
        print("⚠ PyTorch não disponível, usando CPU")
        device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    
    print(f"Usando dispositivo: {device}")
    print(f"Treinando com {NUM_ENVS} ambientes paralelos.")
    
    # ==========
    # Inicializar sistemas de curriculum adaptativo e self-play (sempre ativados)
    # ==========
    adaptive_curriculum = AdaptiveCurriculum(
        stages=CURRICULUM_STAGES,
        min_win_rate=MIN_WIN_RATE_TO_ADVANCE,
        min_episodes=MIN_EPISODES_FOR_EVAL,
        window_size=WINDOW_SIZE,
    )
    print("✓ Curriculum adaptativo ativado")
    
    self_play_manager = SelfPlayManager(
        checkpoint_dir=CHECKPOINT_DIR_BASE,
        max_checkpoints=SELF_PLAY_POOL_SIZE,
        strategy="balanced",
    )
    print(f"✓ Self-play ativado (ratio: {SELF_PLAY_RATIO:.0%})")
    
    # ==========
    # Loop de curriculum (adaptativo ou fixo)
    # ==========
    model: Optional[PPO] = None

    # Loop principal de treinamento
    stage_idx = 0
    while stage_idx < len(CURRICULUM_STAGES):
        # Obter estágio atual (sempre adaptativo)
        stage = adaptive_curriculum.get_current_stage()
        stage_idx = adaptive_curriculum.current_stage_idx
        
        stage_name = stage["name"]
        level = stage["level"]
        total_timesteps = stage["total_timesteps"]

        # Diretórios específicos do estágio
        log_dir = os.path.join(LOG_DIR_BASE, stage_name)
        checkpoint_dir = os.path.join(CHECKPOINT_DIR_BASE, stage_name)
        eval_dir = os.path.join(EVAL_DIR_BASE, stage_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        print(
            f"\n=== Estágio {stage_idx+1}/{len(CURRICULUM_STAGES)}: "
            f"{stage_name} (level={level}, timesteps={total_timesteps:,}) ==="
        )
        
        # Criar run separado do wandb para este estágio
        if wandb_run_stage is not None:
            # Finalizar run anterior se existir
            try:
                wandb_run_stage.finish()
            except:
                pass
        
        wandb_run_stage_name = f"{wandb_run_name_base}_{stage_name}"
        wandb_run_stage = wandb.init(
            project=wandb_project,
            name=wandb_run_stage_name,
            config={
                **wandb_config,
                "stage_name": stage_name,
                "stage_level": level,
                "stage_idx": stage_idx,
                "stage_total_timesteps": total_timesteps,
            },
            sync_tensorboard=True,
            mode=wandb_mode,
            tags=["curriculum_stage", stage_name],
            group=wandb_run_name_base,  # Agrupar todos os estágios sob o mesmo grupo
            reinit=True,  # Permitir múltiplos runs no mesmo processo
        )
        
        if wandb_run_stage is not None:
            print(f"✓ wandb run criado para estágio '{stage_name}'")
            print(f"  Run: {wandb_run_stage_name}")
            print(f"  URL: {wandb_run_stage.url if hasattr(wandb_run_stage, 'url') else 'N/A'}")
        
        # Mostrar dificuldade atual do curriculum adaptativo
        difficulty = adaptive_curriculum.get_current_difficulty()
        print(
            f"  Dificuldade atual: {difficulty['win_rate']:.1%} vitória "
            f"({difficulty['episodes']} episódios)"
        )

        # Ambientes do estágio
        # Se self-play estiver ativo, alguns ambientes podem usar oponentes
        def _make_env_for_vec():
            # Decidir se usa self-play para este ambiente (sempre ativo)
            use_self_play = self_play_manager.should_use_self_play(stage_name)
            opponent_path = None
            
            if use_self_play:
                opponent_path = self_play_manager.select_opponent()
            
            if use_self_play and opponent_path:
                return make_env_with_opponent(level=level, log_dir=log_dir, opponent_model_path=opponent_path)
            else:
                return make_env(level=level, log_dir=log_dir)

        env = make_vec_env(_make_env_for_vec, n_envs=NUM_ENVS)
        
        # Normalizar recompensas para estabilidade (importante com recompensas densas)
        if USE_VEC_NORMALIZE:
            env = VecNormalize(
                env,
                norm_obs=False,  # Observações já são imagens normalizadas
                norm_reward=True,  # NORMALIZAR RECOMPENSAS - crucial para estabilidade!
                clip_obs=10.0,
                clip_reward=10.0,  # Limitar recompensas extremas
                gamma=GAMMA,  # Usar mesmo gamma para consistência
            )
            print(f"✓ VecNormalize aplicado para normalização de recompensas")
        
        eval_env = make_env(level=level, log_dir=log_dir)

        # Callbacks de checkpoint, avaliação e logging de estatísticas
        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_gfootball_repro_{stage_name}",
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=eval_dir,
            eval_freq=CHECKPOINT_FREQ,
            deterministic=True,
            render=False,
        )

        match_stats_callback = MatchStatsCallback(
            stage_name=stage_name,
            print_freq_episodes=10,
            verbose=0,
            wandb_run=wandb_run_stage,  # Usar run específico do estágio
        )

        callbacks: List[BaseCallback] = [
            checkpoint_callback,
            eval_callback,
            match_stats_callback,
        ]

        # Obter hiperparâmetros específicos do estágio
        stage_hyperparams = get_stage_hyperparams(stage_name)
        stage_lr = stage_hyperparams["learning_rate"]
        stage_ent_coef = stage_hyperparams["ent_coef"]
        stage_clip_range = stage_hyperparams["clip_range"]
        stage_n_epochs = stage_hyperparams["n_epochs"]
        
        print(f"  Hiperparâmetros do estágio '{stage_name}' (OTIMIZADOS PARA ESTABILIDADE):")
        print(f"    Learning rate: {stage_lr:.6f} (padrão: {LR:.6f}) - REDUZIDO para estabilidade")
        print(f"    Entropy coef: {stage_ent_coef:.6f} (padrão: {ENT_COEF:.6f})")
        print(f"    Clip range: {stage_clip_range:.3f} (padrão: {CLIP_RANGE:.3f})")
        print(f"    N epochs: {stage_n_epochs} (padrão: {N_EPOCHS}) - REDUZIDO para estabilidade")
        print(f"    Max grad norm: {MAX_GRAD_NORM:.2f} - REDUZIDO para evitar updates grandes")
        print(f"    VecNormalize: {'ATIVADO' if USE_VEC_NORMALIZE else 'DESATIVADO'} - Normalização de recompensas")
        
        # Cria o modelo no primeiro estágio; nos demais, reaproveita e continua treinando
        if model is None:
            # Calcular batch_size a partir de NUM_ENVS agora definido
            batch_size = (N_STEPS * NUM_ENVS) // N_MINIBATCHES
            print(f"  Batch size: {batch_size} (N_STEPS={N_STEPS} * NUM_ENVS={NUM_ENVS} / N_MINIBATCHES={N_MINIBATCHES})")
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=stage_lr,
                n_steps=N_STEPS,
                batch_size=batch_size,
                n_epochs=stage_n_epochs,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=stage_clip_range,
                ent_coef=stage_ent_coef,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                verbose=1,
                tensorboard_log=LOG_DIR_BASE,
                device=device,
            )
        else:
            # Atualiza o ambiente e continua o treinamento (continual learning)
            model.set_env(env)
            
            # Atualizar hiperparâmetros do modelo se necessário
            # Nota: stable-baselines3 não permite mudar todos os hiperparâmetros após criação
            # Mas podemos ajustar learning_rate via schedule
            if hasattr(model, "lr_schedule"):
                # Tentar atualizar learning rate se possível
                try:
                    # Criar novo schedule com o learning rate do estágio
                    from stable_baselines3.common.utils import get_linear_fn
                    model.lr_schedule = get_linear_fn(stage_lr, stage_lr)
                except:
                    pass

        # Treinar neste estágio
        reset_num_timesteps = stage_idx == 0
        
        # Usar o total_timesteps do estágio atual, mas limitar ao máximo de segurança
        stage_max_timesteps = min(total_timesteps, MAX_TIMESTEPS_PER_STAGE)
        
        print(
            f"Iniciando treinamento do estágio '{stage_name}' "
            f"(treinará até atingir critérios ou máximo de {stage_max_timesteps:,} timesteps)..."
        )
        
        # Treinar em chunks menores para permitir avaliação adaptativa frequente
        timesteps_trained = 0
        chunk_size = EVAL_CHUNK_SIZE  # Chunks de 100k para avaliação mais frequente
        chunk_count = 0
        
        # Loop de treinamento: continua até critérios serem atingidos ou limite máximo
        max_retries_per_chunk = 3
        consecutive_failures = 0
        env_recreation_interval = 5  # REDUZIDO: Recriar ambiente a cada 5 chunks (antes era 10)
        critical_chunks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Chunks críticos para recriação
        
        while timesteps_trained < stage_max_timesteps:
            current_chunk = min(chunk_size, stage_max_timesteps - timesteps_trained)
            
            print(f"  Chunk {chunk_count + 1}: Treinando {current_chunk:,} timesteps "
                  f"(total: {timesteps_trained:,}/{stage_max_timesteps:,})")
            
            # RECRIAÇÃO PERIÓDICA DE AMBIENTE para prevenir corrupção de memória
            # Recriar a cada N chunks OU em chunks críticos específicos
            should_recreate = (
                (chunk_count > 0 and chunk_count % env_recreation_interval == 0) or
                (chunk_count in critical_chunks)
            )
            
            if should_recreate:
                print(f"  🔄 Recriação preventiva de ambiente (chunk {chunk_count}) para evitar corrupção...")
                import gc
                import time
                try:
                    env.close()
                    del env
                    gc.collect()
                    time.sleep(2.0)  # AUMENTADO: Dar mais tempo para liberar recursos (2s em vez de 1s)
                except Exception as close_error:
                    print(f"  ⚠ Erro ao fechar ambiente (ignorando): {close_error}")
                    # Limpar mesmo com erro
                    try:
                        gc.collect()
                        time.sleep(1.0)
                    except:
                        pass
                
                # Recriar ambiente
                        try:
                            env = make_vec_env(_make_env_for_vec, n_envs=NUM_ENVS)
                            # Reaplicar normalização se estava habilitada
                            if USE_VEC_NORMALIZE:
                                env = VecNormalize(
                                    env,
                                    norm_obs=False,
                                    norm_reward=True,
                                    clip_obs=10.0,
                                    clip_reward=10.0,
                                    gamma=GAMMA,
                                )
                            model.set_env(env)
                            gc.collect()  # Limpar após recriação
                            print(f"  ✓ Ambiente recriado com sucesso")
                        except Exception as recreate_err:
                            print(f"  ✗ Erro ao recriar ambiente: {recreate_err}")
                            # Tentar novamente após mais tempo
                            time.sleep(3.0)
                            gc.collect()
                            env = make_vec_env(_make_env_for_vec, n_envs=NUM_ENVS)
                            # Reaplicar normalização
                            if USE_VEC_NORMALIZE:
                                env = VecNormalize(
                                    env,
                                    norm_obs=False,
                                    norm_reward=True,
                                    clip_obs=10.0,
                                    clip_reward=10.0,
                                    gamma=GAMMA,
                                )
                            model.set_env(env)
                            print(f"  ✓ Ambiente recriado na segunda tentativa")
            
            # Tentar treinar com retry logic
            chunk_success = False
            for retry_attempt in range(max_retries_per_chunk):
                try:
                    # Salvar checkpoint antes de cada chunk para recovery
                    if chunk_count > 0 and chunk_count % 5 == 0:
                        recovery_path = os.path.join(
                            checkpoint_dir, 
                            f"ppo_gfootball_repro_{stage_name}_recovery_chunk_{chunk_count}"
                        )
                        print(f"  💾 Salvando checkpoint de recovery em: {recovery_path}")
                        model.save(recovery_path)
                    
                    # Limpeza de memória antes do treinamento
                    if chunk_count > 0 and chunk_count % 3 == 0:
                        import gc
                        gc.collect()
                    
                    model.learn(
                        total_timesteps=current_chunk,
                        callback=callbacks,
                        tb_log_name=f"ppo_gfootball_repro_{stage_name}",
                        reset_num_timesteps=reset_num_timesteps and timesteps_trained == 0,
                    )
                    
                    # Se chegou aqui, treinamento foi bem-sucedido
                    timesteps_trained += current_chunk
                    chunk_count += 1
                    consecutive_failures = 0
                    chunk_success = True
                    break
                    
                except KeyboardInterrupt:
                    print("\n⚠ Interrupção pelo usuário detectada. Salvando modelo...")
                    recovery_path = os.path.join(
                        checkpoint_dir, 
                        f"ppo_gfootball_repro_{stage_name}_interrupted_chunk_{chunk_count}"
                    )
                    model.save(recovery_path)
                    print(f"  ✓ Modelo salvo em: {recovery_path}")
                    raise
                    
                except Exception as e:
                    consecutive_failures += 1
                    error_msg = str(e)
                    error_type = type(e).__name__
                    
                    print(f"\n⚠ ERRO durante treinamento (tentativa {retry_attempt + 1}/{max_retries_per_chunk}):")
                    print(f"  Tipo: {error_type}")
                    print(f"  Mensagem: {error_msg}")
                    
                    # Verificar se é erro crítico de segmentação
                    if "signal 11" in error_msg.lower() or "segfault" in error_msg.lower() or "SIGSEGV" in error_msg.upper():
                        print("  ⚠ Erro de segmentação detectado (SIGSEGV)")
                        
                        # Tentar salvar modelo antes de recriar ambiente
                        try:
                            recovery_path = os.path.join(
                                checkpoint_dir, 
                                f"ppo_gfootball_repro_{stage_name}_segfault_recovery_chunk_{chunk_count}"
                            )
                            print(f"  💾 Tentando salvar recovery em: {recovery_path}")
                            model.save(recovery_path)
                            print(f"  ✓ Recovery salvo com sucesso")
                        except Exception as save_error:
                            print(f"  ✗ Erro ao salvar recovery: {save_error}")
                        
                        # Recriar ambiente para evitar corrupção
                        print("  🔄 Recriando ambiente para evitar corrupção...")
                        import gc
                        import time
                        try:
                            env.close()
                            del env
                            gc.collect()
                            time.sleep(2.0)  # Dar mais tempo para liberar recursos após erro
                        except Exception as close_err:
                            print(f"  ⚠ Erro ao fechar ambiente: {close_err}")
                        
                        # Recriar ambiente
                        try:
                            env = make_vec_env(_make_env_for_vec, n_envs=NUM_ENVS)
                            # Reaplicar normalização se estava habilitada
                            if USE_VEC_NORMALIZE:
                                env = VecNormalize(
                                    env,
                                    norm_obs=False,
                                    norm_reward=True,
                                    clip_obs=10.0,
                                    clip_reward=10.0,
                                    gamma=GAMMA,
                                )
                            model.set_env(env)
                            print("  ✓ Ambiente recriado com sucesso")
                            gc.collect()  # Limpar após recriação
                        except Exception as recreate_err:
                            print(f"  ✗ Erro ao recriar ambiente: {recreate_err}")
                            raise
                    
                    if retry_attempt < max_retries_per_chunk - 1:
                        print(f"  ⏳ Aguardando 5 segundos antes de retentar...")
                        import time
                        time.sleep(5)
                    else:
                        # Todas as tentativas falharam
                        print(f"\n❌ FALHA CRÍTICA: Não foi possível treinar após {max_retries_per_chunk} tentativas")
                        print(f"  Consecutive failures: {consecutive_failures}")
                        
                        # Salvar último estado conhecido
                        try:
                            emergency_path = os.path.join(
                                checkpoint_dir, 
                                f"ppo_gfootball_repro_{stage_name}_emergency_chunk_{chunk_count}"
                            )
                            print(f"  💾 Salvando estado de emergência em: {emergency_path}")
                            model.save(emergency_path)
                        except Exception as save_error:
                            print(f"  ✗ Erro ao salvar estado de emergência: {save_error}")
                        
                        # Se muitas falhas consecutivas, reduzir tamanho do chunk
                        if consecutive_failures >= 3:
                            print(f"  ⚠ Muitas falhas consecutivas. Reduzindo tamanho do chunk...")
                            chunk_size = max(chunk_size // 2, 10000)  # Reduzir pela metade, mínimo 10k
                            print(f"  Novo chunk_size: {chunk_size:,}")
                            consecutive_failures = 0
                            
                            # Recriar ambiente novamente com limpeza agressiva
                            import gc
                            import time
                            try:
                                env.close()
                                del env
                                gc.collect()
                                time.sleep(2.0)
                            except:
                                pass
                            try:
                                env = make_vec_env(_make_env_for_vec, n_envs=NUM_ENVS)
                                # Reaplicar normalização se estava habilitada
                                if USE_VEC_NORMALIZE:
                                    env = VecNormalize(
                                        env,
                                        norm_obs=False,
                                        norm_reward=True,
                                        clip_obs=10.0,
                                        clip_reward=10.0,
                                        gamma=GAMMA,
                                    )
                                model.set_env(env)
                                gc.collect()
                            except Exception as recreate_err:
                                print(f"  ✗ Erro ao recriar ambiente: {recreate_err}")
                                raise
                            continue
                        else:
                            # Re-lançar exceção se não é falha consecutiva demais
                            raise
            
            if not chunk_success:
                print(f"\n❌ Não foi possível completar o chunk após todas as tentativas.")
                print(f"  Progresso salvo: {timesteps_trained:,} timesteps treinados")
                break  # Sair do loop de treinamento deste estágio
            
            # Atualizar métricas do curriculum adaptativo (sempre ativo)
            stats = match_stats_callback.get_match_stats()
            if stats:
                # Converter stats para formato esperado
                for match in match_stats_callback.recent_matches:
                    adaptive_curriculum.update_metrics(stage_name, match)
                
                # Verificar se deve avançar
                if adaptive_curriculum.should_advance(stage_name):
                    adaptive_curriculum.advance()
                    print(f"✓ Estágio '{stage_name}' concluído após {timesteps_trained:,} timesteps")
                    break  # Sair do loop e ir para próximo estágio
                
                # Verificar se deve regredir (agora ativado com critérios melhorados)
                if adaptive_curriculum.should_regress(stage_name, chunk_count):
                    adaptive_curriculum.regress()
                    print(f"← Regredindo após {timesteps_trained:,} timesteps")
                    break  # Sair do loop e voltar ao estágio anterior
                
                # Logar progresso do curriculum e self-play (sempre ativo)
                try:
                    difficulty = adaptive_curriculum.get_current_difficulty()
                    stability_count = adaptive_curriculum.stability_tracker.get(stage_name, 0)
                    regression_count = adaptive_curriculum.regression_tracker.get(stage_name, 0)
                    
                    log_dict = {
                        "curriculum/current_stage": stage_idx,
                        "curriculum/stage_name": stage_name,
                        "curriculum/win_rate": difficulty["win_rate"],
                        "curriculum/episodes": difficulty["episodes"],
                        "curriculum/recent_episodes": difficulty["recent_episodes"],
                        "curriculum/stability_count": stability_count,
                        "curriculum/regression_count": regression_count,
                        "curriculum/timesteps_trained": timesteps_trained,
                        "curriculum/chunk_count": chunk_count,
                        "self_play/opponent_pool_size": self_play_manager.get_pool_size(),
                        "self_play/enabled": ENABLE_SELF_PLAY,
                        "self_play/ratio": SELF_PLAY_RATIO,
                    }
                    
                    # Adicionar métricas de diferença de gols se disponíveis
                    if stats:
                        log_dict.update({
                            "curriculum/goal_difference_mean": stats.get("goal_difference_mean", 0),
                            "curriculum/win_by_1plus_goals_rate": stats.get("win_by_1plus_goals_rate", 0),
                            "curriculum/avg_goals_scored": stats.get("avg_goals_scored", 0),
                            "curriculum/avg_goals_conceded": stats.get("avg_goals_conceded", 0),
                        })
                    
                    # Logar no run do estágio e no run principal
                    if wandb_run_stage is not None:
                        wandb_run_stage.log(log_dict, step=model.num_timesteps)
                    if wandb_run_main is not None:
                        wandb_run_main.log(log_dict, step=model.num_timesteps)
                except Exception as e:
                    if chunk_count % 10 == 0:  # Log erro ocasionalmente
                        print(f"⚠ Erro ao logar no wandb: {e}")
            
            # Verificar se atingiu o limite máximo (safety net)
            if timesteps_trained >= stage_max_timesteps:
                print(f"⚠ Limite máximo de timesteps atingido para '{stage_name}' "
                      f"({stage_max_timesteps:,}). Avançando mesmo sem atingir critérios.")
                break

        # Salvar um snapshot ao final de cada estágio
        stage_final_path = os.path.join(
            checkpoint_dir, f"ppo_gfootball_repro_{stage_name}_final"
        )
        print(f"Salvando modelo do estágio em {stage_final_path}")
        model.save(stage_final_path)
        
        # Adicionar checkpoint ao pool de self-play (sempre ativo)
        stats = match_stats_callback.get_match_stats()
        self_play_manager.add_checkpoint(
            stage_final_path,
            stats,
            model.num_timesteps,
        )
        print(f"  Pool de self-play: {self_play_manager.get_pool_size()} checkpoints")

        # Notificar wandb do fim do estágio (sempre ativo)
        try:
            stats = match_stats_callback.get_match_stats()
            difficulty = adaptive_curriculum.get_current_difficulty()
            stability_count = adaptive_curriculum.stability_tracker.get(stage_name, 0)
            
            log_dict = {
                "stage_completed": stage_name,
                "stage_total_timesteps": timesteps_trained,
                "stage_chunks": chunk_count,
                "curriculum/final_win_rate": difficulty["win_rate"],
                "curriculum/final_episodes": difficulty["episodes"],
                "curriculum/final_stability_count": stability_count,
                "self_play/final_pool_size": self_play_manager.get_pool_size(),
            }
            if stats:
                log_dict.update({
                    f"stage_final/{k}": v for k, v in stats.items()
                })
                # Adicionar métricas específicas de diferença de gols
                log_dict.update({
                    "stage_final/goal_difference_mean": stats.get("goal_difference_mean", 0),
                    "stage_final/goal_difference_std": stats.get("goal_difference_std", 0),
                    "stage_final/win_by_1plus_goals_rate": stats.get("win_by_1plus_goals_rate", 0),
                })
            
            # Logar no run do estágio e no run principal
            if wandb_run_stage is not None:
                wandb_run_stage.log(log_dict, step=model.num_timesteps)
            if wandb_run_main is not None:
                wandb_run_main.log(log_dict, step=model.num_timesteps)
        except Exception as e:
            print(f"⚠ Aviso: Erro ao logar fim do estágio no wandb: {e}")
        
        # Avançar para próximo estágio se não avançou automaticamente
        if adaptive_curriculum.current_stage_idx == stage_idx:
            # Se ainda está no mesmo estágio, avançar manualmente
            stage_idx += 1
            if stage_idx < len(CURRICULUM_STAGES):
                adaptive_curriculum.current_stage_idx = stage_idx

    # Salvar modelo final global após todos os estágios
    final_path = os.path.join(
        CHECKPOINT_DIR_BASE, "ppo_gfootball_repro_curriculum_final"
    )
    print(f"\nCurriculum concluído! Salvando modelo final em {final_path}")
    model.save(final_path)

    # Finalizar wandb (sempre ativo)
    try:
        # Finalizar run do último estágio se ainda estiver ativo
        if wandb_run_stage is not None:
            wandb_run_stage.finish()
            print("✓ wandb run do estágio finalizado com sucesso!")
        # Finalizar run principal
        if wandb_run_main is not None:
            wandb_run_main.finish()
            print("✓ wandb run principal finalizado com sucesso!")
    except Exception as e:
        print(f"⚠ Aviso: Erro ao finalizar wandb: {e}")

    print("Treinamento concluído!")


if __name__ == "__main__":
    main()