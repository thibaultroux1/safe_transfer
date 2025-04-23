from gymnasium.envs.registration import register


register(
    id='Drone2D-v0',
    entry_point='safe_transfer.environments.drone_2d:Drone2dEnv',
)

register(
    id='Drone2D-budget-v0',
    entry_point='safe_transfer.environments.drone_2d_budget:Drone2dBudgetEnv'
)

register(
    id='Drone2D-budget-package-v0',
    entry_point='safe_transfer.environments.drone_2d_budget_package:Drone2dBudgetPackageEnv'
)

register(
    id='Drone2D-budget-hierarchical-v0',
    entry_point='safe_transfer.environments.drone_2d_hierarchical:Drone2dBudgetHierarchyEnv'
)

register(
    id='Drone2D-budget-package-hierarchical-v0',
    entry_point='safe_transfer.environments.drone_2d_hierarchical:Drone2dBudgetPackageHierarchyEnv'
)