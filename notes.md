# Dictionary keys for obs.observation
    single_select
    multi_select
    build_queue
    cargo
    cargo_slots_available
    feature_screen
    feature_minimap
    last_actions
    action_result
    alerts
    game_loop
    score_cumulative
    score_by_category
    score_by_vital
    player
    control_groups
    available_actions

# All of the stuff in BaseAgent.action_spec
    ```ValidActions(
    types=Arguments(
        screen=ArgumentType(id=0, name='screen', sizes=(84, 84), fn=None, values=None),
        minimap=ArgumentType(id=1, name='minimap', sizes=(64, 64), fn=None, values=None), 
        screen2=ArgumentType(id=2, name='screen2', sizes=(84, 84), fn=None, values=None), 
        queued=ArgumentType(id=3, name='queued', sizes=(2,), fn=None, values=None), 
        control_group_act=ArgumentType(id=4, name='control_group_act', sizes=(5,), fn=None, values=None), 
        control_group_id=ArgumentType(id=5, name='control_group_id', sizes=(10,), fn=None, values=None), 
        select_point_act=ArgumentType(id=6, name='select_point_act', sizes=(4,), fn=None, values=None), 
        select_add=ArgumentType(id=7, name='select_add', sizes=(2,), fn=None, values=None), 
        select_unit_act=ArgumentType(id=8, name='select_unit_act', sizes=(4,), fn=None, values=None), 
        select_unit_id=ArgumentType(id=9, name='select_unit_id', sizes=(500,), fn=None, values=None), 
        select_worker=ArgumentType(id=10, name='select_worker', sizes=(4,), fn=None, values=None), 
        build_queue_id=ArgumentType(id=11, name='build_queue_id', sizes=(10,), fn=None, values=None), 
        unload_id=ArgumentType(id=12, name='unload_id', sizes=(500,), fn=None, values=None)), 
        functions=<pysc2.lib.actions.Functions object at 0x117739e48>)```