from DHTable import DHTable
from ArmKinematics import ArmKinematics
import math

joint_angles = [math.pi, math.pi, 2*(math.pi), -(math.pi), 0, 0]

table1 = DHTable(joint_angles)

kinematics = ArmKinematics(table1)

mat, x,y,z = kinematics.endEffectorPose()
