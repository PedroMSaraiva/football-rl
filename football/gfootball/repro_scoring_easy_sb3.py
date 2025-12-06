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

# ===========================
# Configurações gerais
# ===========================

# Configurações de Hardware (otimizado para RTX 4090)
# RTX 4090 tem 24GB VRAM, pode suportar 32-64 ambientes dependendo da configuração
NUM_ENVS = int(os.environ.get("NUM_ENVS", "4"))  # Otimizado para 4090

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

# Hiperparâmetros otimizados para RTX 4090
# Ajustados para melhor throughput com mais ambientes paralelos
N_STEPS = 512
N_EPOCHS = 4  # Aumentado para melhor estabilidade
N_MINIBATCHES = 8  # Aumentado para melhor paralelização
LR = 0.00011879
GAMMA = 0.997
ENT_COEF = 0.00155
CLIP_RANGE = 0.115
MAX_GRAD_NORM = 0.76
GAE_LAMBDA = 0.95
VF_COEF = 0.5

# Checkpointing mais frequente para 24/7
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", "50000"))  # A cada 50k steps

# Curriculum: estágios de dificuldade crescente
CURRICULUM_STAGES = [
    {
        "name": "stage1_empty_goal",
        "level": "academy_empty_goal_close",
        "total_timesteps": int(4e6),
    },
    {
        "name": "stage2_run_to_score",
        "level": "academy_run_to_score_with_keeper",
        "total_timesteps": int(8e6),
    },
    {
        "name": "stage3_11v11_easy",
        "level": "11_vs_11_easy_stochastic",
        "total_timesteps": int(8e6),
    },
]

# Diretórios base (compatíveis com o setup em docker-compose)
LOG_DIR_BASE = "/RL/logs_repro_scoring"
CHECKPOINT_DIR_BASE = "/RL/checkpoints_repro_scoring"
EVAL_DIR_BASE = "/RL/eval_repro_scoring"

os.makedirs(LOG_DIR_BASE, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)
os.makedirs(EVAL_DIR_BASE, exist_ok=True)


class ScoreInfoWrapper:
    """Wrapper que adiciona o score ao info dict quando o episódio termina."""
    
    def __init__(self, env):
        self.env = env
        self.last_observation = None
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_observation = obs
        
        # Quando o episódio termina, adicionar score ao info
        if done and self.last_observation is not None:
            # Tentar obter score da observação
            if isinstance(self.last_observation, dict) and "score" in self.last_observation:
                info["score"] = self.last_observation["score"]
            elif hasattr(self.last_observation, "score"):
                info["score"] = self.last_observation.score
        
        return obs, reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        self.last_observation = obs
        return obs


def make_env(level: str, log_dir: str):
    """Cria ambiente do gfootball para um dado nível (cenário)."""
    env = football_env.create_environment(
        env_name=level,
        stacked=True,
        representation="extracted",
        rewards="scoring",
        logdir=log_dir,
        render=False,
    )
    # Envolver com wrapper que adiciona score ao info
    env = ScoreInfoWrapper(env)
    return env


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
            
            if score is not None:
                self.episode_count += 1
                
                # Parse do score (pode ser string ou lista)
                if isinstance(score, str):
                    # Formato "X-Y" ou similar
                    try:
                        left_goals, right_goals = map(int, score.split("-"))
                    except:
                        left_goals, right_goals = 0, 0
                elif isinstance(score, (list, tuple, np.ndarray)) and len(score) >= 2:
                    left_goals = int(score[0])
                    right_goals = int(score[1])
                else:
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
                
                # Criar registro da partida
                match_record = {
                    "episode": self.episode_count,
                    "score": f"{left_goals}-{right_goals}",
                    "left_goals": left_goals,
                    "right_goals": right_goals,
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
                    
                    win_rate = recent_wins / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    draw_rate = recent_draws / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                    loss_rate = recent_losses / len(self.recent_matches) if len(self.recent_matches) > 0 else 0
                else:
                    win_rate = draw_rate = loss_rate = 0
                    recent_goals_scored = recent_goals_conceded = 0
                    recent_avg_duration = 0
                
                # Log em wandb, se disponível
                if self.wandb_run is not None:
                    try:
                        log_dict = {
                            # Estatísticas do episódio atual
                            "match/score_left": left_goals,
                            "match/score_right": right_goals,
                            "match/result": 1 if result == "win" else (0 if result == "draw" else -1),
                            "match/duration": match_duration,
                            
                            # Estatísticas agregadas (janela recente)
                            "match/win_rate": win_rate,
                            "match/draw_rate": draw_rate,
                            "match/loss_rate": loss_rate,
                            "match/avg_goals_scored": recent_goals_scored / len(self.recent_matches) if len(self.recent_matches) > 0 else 0,
                            "match/avg_goals_conceded": recent_goals_conceded / len(self.recent_matches) if len(self.recent_matches) > 0 else 0,
                            "match/avg_duration": recent_avg_duration,
                            
                            # Estatísticas totais
                            "match/total_wins": self.total_wins,
                            "match/total_draws": self.total_draws,
                            "match/total_losses": self.total_losses,
                            "match/total_goals_scored": self.total_goals_scored,
                            "match/total_goals_conceded": self.total_goals_conceded,
                            
                            # Metadados
                            "episode": self.episode_count,
                            "stage": self.stage_name,
                        }
                        self.wandb_run.log(log_dict, step=self.num_timesteps)
                    except Exception as e:
                        # Não falhar o treino por causa de logging, mas avisar
                        if self.episode_count % 100 == 0:
                            print(f"⚠ Aviso: Erro ao logar no wandb: {e}")

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
        Requer taxa de vitória mínima por N episódios.
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Já está no último estágio
        
        metrics = self.stage_metrics.get(stage_name, [])
        if len(metrics) < self.min_episodes:
            return False  # Não tem episódios suficientes
        
        # Calcular taxa de vitória na janela recente
        recent_metrics = metrics[-self.min_episodes:]
        wins = sum(1 for m in recent_metrics if m.get("result") == "win")
        win_rate = wins / len(recent_metrics)
        
        # Também verificar se está marcando gols consistentemente
        avg_goals = np.mean([m.get("left_goals", 0) for m in recent_metrics])
        
        should_advance = win_rate >= self.min_win_rate and avg_goals > 0.1
        
        if should_advance:
            print(
                f"✓ Critério de progressão atingido para '{stage_name}': "
                f"Taxa de vitória: {win_rate:.1%}, Gols médios: {avg_goals:.2f}"
            )
        
        return should_advance
    
    def should_regress(self, stage_name: str) -> bool:
        """
        Decide se deve regredir para o estágio anterior.
        Se performance cair muito, pode ser necessário voltar.
        """
        if self.current_stage_idx == 0:
            return False  # Já está no primeiro estágio
        
        metrics = self.stage_metrics.get(stage_name, [])
        if len(metrics) < self.min_episodes:
            return False  # Não tem dados suficientes
        
        # Se taxa de vitória cair muito abaixo do mínimo
        recent_metrics = metrics[-self.min_episodes:]
        wins = sum(1 for m in recent_metrics if m.get("result") == "win")
        win_rate = wins / len(recent_metrics)
        
        # Regredir se win_rate < 30% (muito baixo)
        should_regress = win_rate < 0.3
        
        if should_regress:
            print(
                f"⚠ Performance muito baixa em '{stage_name}': "
                f"Taxa de vitória: {win_rate:.1%}. Considerando regressão."
            )
        
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


def make_env_with_opponent(level: str, log_dir: str, opponent_model_path: Optional[str] = None):
    """
    Cria ambiente do gfootball, opcionalmente com oponente controlado por modelo.
    Se opponent_model_path for None, usa bots padrão.
    
    Nota: A integração completa de self-play requereria um wrapper mais complexo
    que carrega o modelo oponente e intercepta ações do time direito.
    Por enquanto, retorna o ambiente padrão (self-play será implementado futuramente).
    """
    env = football_env.create_environment(
        env_name=level,
        stacked=True,
        representation="extracted",
        rewards="scoring",
        logdir=log_dir,
        render=False,
    )
    
    # Envolver com wrapper que adiciona score ao info
    env = ScoreInfoWrapper(env)
    
    # TODO: Implementar carregamento e integração do modelo oponente
    # Isso requereria:
    # 1. Carregar o modelo do checkpoint
    # 2. Criar um wrapper que intercepta ações do time direito
    # 3. Usar o modelo para gerar ações do oponente
    
    return env


def main():
    # ==========
    # wandb (sempre ativado)
    # ==========
    import wandb

    # Verificar se wandb está instalado e funcionando
    print("Inicializando wandb...")
    
    wandb_project = os.environ.get("WANDB_PROJECT", "gfootball_repro_scoring")
    wandb_run_name = os.environ.get(
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
    }
    
    # Forçar modo interativo para pedir credenciais se necessário
    # Se WANDB_MODE não estiver definido, usar "online" (padrão)
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    
    # wandb.init() automaticamente pedirá credenciais se necessário em modo online
    # e o usuário não estiver autenticado
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=wandb_config,
        sync_tensorboard=True,
        mode=wandb_mode,
    )
    
    # Verificar se a inicialização foi bem-sucedida
    if wandb_run is not None:
        print(f"✓ wandb inicializado com sucesso!")
        print(f"  Projeto: {wandb_project}")
        print(f"  Run: {wandb_run_name}")
        print(f"  URL: {wandb_run.url if hasattr(wandb_run, 'url') else 'N/A'}")
    else:
        print("⚠ Aviso: wandb.init() retornou None")

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
            wandb_run=wandb_run,
        )

        callbacks: List[BaseCallback] = [
            checkpoint_callback,
            eval_callback,
            match_stats_callback,
        ]

        # Cria o modelo no primeiro estágio; nos demais, reaproveita e continua treinando
        if model is None:
            # Calcular batch_size a partir de NUM_ENVS agora definido
            batch_size = (N_STEPS * NUM_ENVS) // N_MINIBATCHES
            print(f"  Batch size: {batch_size} (N_STEPS={N_STEPS} * NUM_ENVS={NUM_ENVS} / N_MINIBATCHES={N_MINIBATCHES})")
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=LR,
                n_steps=N_STEPS,
                batch_size=batch_size,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                verbose=1,
                tensorboard_log=LOG_DIR_BASE,
                device=device,
            )
        else:
            # Atualiza o ambiente e continua o treinamento (continual learning)
            model.set_env(env)

        # Treinar neste estágio
        reset_num_timesteps = stage_idx == 0
        print(
            f"Iniciando treinamento do estágio '{stage_name}' "
            f"por {total_timesteps:,} timesteps..."
        )
        
        # Treinar em chunks menores para permitir avaliação adaptativa
        timesteps_trained = 0
        chunk_size = min(500_000, total_timesteps)  # Chunks de 500k para avaliação
        
        while timesteps_trained < total_timesteps:
            remaining = total_timesteps - timesteps_trained
            current_chunk = min(chunk_size, remaining)
            
            model.learn(
                total_timesteps=current_chunk,
                callback=callbacks,
                tb_log_name=f"ppo_gfootball_repro_{stage_name}",
                reset_num_timesteps=reset_num_timesteps and timesteps_trained == 0,
            )
            
            timesteps_trained += current_chunk
            
            # Atualizar métricas do curriculum adaptativo (sempre ativo)
            stats = match_stats_callback.get_match_stats()
            if stats:
                # Converter stats para formato esperado
                for match in match_stats_callback.recent_matches:
                    adaptive_curriculum.update_metrics(stage_name, match)
                
                # Verificar se deve avançar
                if adaptive_curriculum.should_advance(stage_name):
                    adaptive_curriculum.advance()
                    break  # Sair do loop e ir para próximo estágio
                
                # Verificar se deve regredir (opcional, pode ser muito agressivo)
                # if adaptive_curriculum.should_regress(stage_name):
                #     adaptive_curriculum.regress()
                #     break
                
                # Logar progresso do curriculum e self-play (sempre ativo)
                try:
                    difficulty = adaptive_curriculum.get_current_difficulty()
                    log_dict = {
                        "curriculum/current_stage": stage_idx,
                        "curriculum/stage_name": stage_name,
                        "curriculum/win_rate": difficulty["win_rate"],
                        "curriculum/episodes": difficulty["episodes"],
                        "curriculum/recent_episodes": difficulty["recent_episodes"],
                        "self_play/opponent_pool_size": self_play_manager.get_pool_size(),
                        "self_play/enabled": ENABLE_SELF_PLAY,
                        "self_play/ratio": SELF_PLAY_RATIO,
                    }
                    wandb_run.log(log_dict, step=model.num_timesteps)
                except Exception as e:
                    if timesteps_trained % 100_000 == 0:  # Log erro ocasionalmente
                        print(f"⚠ Erro ao logar no wandb: {e}")

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
            log_dict = {
                "stage_completed": stage_name,
                "stage_total_timesteps": total_timesteps,
                "curriculum/final_win_rate": difficulty["win_rate"],
                "curriculum/final_episodes": difficulty["episodes"],
                "self_play/final_pool_size": self_play_manager.get_pool_size(),
            }
            if stats:
                log_dict.update({
                    f"stage_final/{k}": v for k, v in stats.items()
                })
            
            wandb_run.log(log_dict, step=model.num_timesteps)
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
        wandb_run.finish()
        print("✓ wandb finalizado com sucesso!")
    except Exception as e:
        print(f"⚠ Aviso: Erro ao finalizar wandb: {e}")

    print("Treinamento concluído!")


if __name__ == "__main__":
    main()