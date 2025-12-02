import gfootball.env as football_env

env = football_env.create_environment(
    env_name="11_vs_11_stochastic", 
    stacked=True, 
    representation='simple115v2',
    logdir='videos', 
    write_goal_dumps=False, 
    write_full_episode_dumps=True,  # Must be True for videos to be saved 
    render=True,
    write_video=True
)
print("Action Space:")
print(env.action_space)
print("--------------------------------")
print("Observation Space:")
print(env.observation_space)
print("--------------------------------")
env.reset()
steps = 0
while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  steps += 1
  if steps % 100 == 0:
    print(f"Step {steps} Reward: {rew}")
  if done:
    break
env.close()
print(f"Steps: {steps} Reward: {rew}")