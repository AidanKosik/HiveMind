from pysc2.lib import actions


no_op = str(actions._FUNCTIONS[1])
print(no_op)

no_op_id = no_op.split('/')
print(no_op_id[0])
id = int(no_op_id[0])