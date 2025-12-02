#!/usr/bin/env python3
"""
Equivalente do repro_scoring_easy.sh usando stable-baselines3,
com:
  - curriculum learning (cenários de dificuldade crescente)
  - paralelização configurável
  - monitoramento (wandb opcional) e logging de gols/placares.
"""

import os
from typing import List, Optional, Dict, Any

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

# Paralelização configurável via variável de ambiente (ex: NUM_ENVS=32 ...)
NUM_ENVS = int(os.environ.get("NUM_ENVS", "4"))

# Hiperparâmetros baseados no repro_scoring_easy.sh original
N_STEPS = 512
N_EPOCHS = 2
N_MINIBATCHES = 4
LR = 0.00011879
GAMMA = 0.997
ENT_COEF = 0.00155
CLIP_RANGE = 0.115
MAX_GRAD_NORM = 0.76
GAE_LAMBDA = 0.95
VF_COEF = 0.5

# Curriculum: estágios de dificuldade crescente
CURRICULUM_STAGES = [
    {
        "name": "stage1_empty_goal",
        "level": "academy_empty_goal_close",
        "total_timesteps": int(1e6),
    },
    {
        "name": "stage2_run_to_score",
        "level": "academy_run_to_score_with_keeper",
        "total_timesteps": int(2e6),
    },
    {
        "name": "stage3_11v11_easy",
        "level": "11_vs_11_easy_stochastic",
        "total_timesteps": int(2e6),
    },
]

# Diretórios base (compatíveis com o setup em docker-compose)
LOG_DIR_BASE = "/RL/logs_repro_scoring"
CHECKPOINT_DIR_BASE = "/RL/checkpoints_repro_scoring"
EVAL_DIR_BASE = "/RL/eval_repro_scoring"

os.makedirs(LOG_DIR_BASE, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)
os.makedirs(EVAL_DIR_BASE, exist_ok=True)


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
    return env


class ScoreLoggingCallback(BaseCallback):
    """
    Callback para:
      - imprimir e registrar placares/gols por episódio
      - logar em wandb se estiver ativado.
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
        self.episode_scores: List[str] = []

    def _on_step(self) -> bool:
        # infos é uma lista (um por ambiente) em ambientes vetorizados
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, done in enumerate(dones):
            if not done:
                continue
            info = infos[idx] if idx < len(infos) else {}
            score = info.get("score")
            if score is not None:
                self.episode_count += 1
                self.episode_scores.append(str(score))

                # Log em wandb, se disponível
                if self.wandb_run is not None:
                    try:
                        self.wandb_run.log(
                            {
                                "episode_score": score,
                                "episode": self.episode_count,
                                "stage": self.stage_name,
                            },
                            step=self.num_timesteps,
                        )
                    except Exception:
                        # Não falhar o treino por causa de logging
                        pass

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

        return True


def main():
    # ==========
    # wandb (opcional)
    # ==========
    wandb_run = None
    use_wandb = os.environ.get("USE_WANDB", "1") == "1"
    if use_wandb:
        try:
            import wandb

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
            }
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config,
                sync_tensorboard=True,
            )
        except ImportError:
            print(
                "wandb não está instalado; "
                "defina USE_WANDB=0 ou instale com `pip install wandb`."
            )
            wandb_run = None

    # ==========
    # Loop de curriculum
    # ==========
    model: Optional[PPO] = None

    device = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    print(f"Usando dispositivo: {device}")
    print(f"Treinando com {NUM_ENVS} ambientes paralelos.")

    for i, stage in enumerate(CURRICULUM_STAGES):
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
            f"\n=== Estágio {i+1}/{len(CURRICULUM_STAGES)}: "
            f"{stage_name} (level={level}, timesteps={total_timesteps:,}) ==="
        )

        # Ambientes do estágio
        def _make_env_for_vec():
            return make_env(level=level, log_dir=log_dir)

        env = make_vec_env(_make_env_for_vec, n_envs=NUM_ENVS)
        eval_env = make_env(level=level, log_dir=log_dir)

        # Callbacks de checkpoint, avaliação e logging de gols
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_gfootball_repro_{stage_name}",
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=eval_dir,
            eval_freq=100_000,
            deterministic=True,
            render=False,
        )

        score_logging_callback = ScoreLoggingCallback(
            stage_name=stage_name,
            print_freq_episodes=10,
            verbose=0,
            wandb_run=wandb_run,
        )

        callbacks: List[BaseCallback] = [
            checkpoint_callback,
            eval_callback,
            score_logging_callback,
        ]

        # Cria o modelo no primeiro estágio; nos demais, reaproveita e continua treinando
        if model is None:
            # Calcular batch_size a partir de NUM_ENVS agora definido
            batch_size = (N_STEPS * NUM_ENVS) // N_MINIBATCHES
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
        reset_num_timesteps = i == 0
        print(
            f"Iniciando treinamento do estágio '{stage_name}' "
            f"por {total_timesteps:,} timesteps..."
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"ppo_gfootball_repro_{stage_name}",
            reset_num_timesteps=reset_num_timesteps,
        )

        # Salvar um snapshot ao final de cada estágio
        stage_final_path = os.path.join(
            checkpoint_dir, f"ppo_gfootball_repro_{stage_name}_final"
        )
        print(f"Salvando modelo do estágio em {stage_final_path}")
        model.save(stage_final_path)

        # Notificar wandb do fim do estágio
        if wandb_run is not None:
            try:
                wandb_run.log(
                    {
                        "stage_completed": stage_name,
                        "stage_total_timesteps": total_timesteps,
                    }
                )
            except Exception:
                pass

    # Salvar modelo final global após todos os estágios
    final_path = os.path.join(
        CHECKPOINT_DIR_BASE, "ppo_gfootball_repro_curriculum_final"
    )
    print(f"\nCurriculum concluído! Salvando modelo final em {final_path}")
    model.save(final_path)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print("Treinamento concluído!")


if __name__ == "__main__":
    main()