import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FDM():
    """
    環境に関する定義を入れる。
    """
    def __init__(self):
        pass

    def U_wall(self,subject_position,center_position,):
        # 上に凸なポテンシャル場（→斥力）の計算式
        coef_x=1.0
        coef_y=1.0
        potential=coef_x*(subject_position[0]-center_position[0])**2+coef_y*(subject_position[1]-center_position[1])**2
        return potential

    def W_human(self,subject_position,center_position,):
        # 下に凸なポテンシャル場（→引力）の計算式
        coef_x=1.0
        coef_y=1.0
        potential=-coef_x*(subject_position[0]-center_position[0])**2+-coef_y*(subject_position[1]-center_position[1])**2
        return potential


class Human(FDM):
    """
    人間位置・行動に関する変数を管理
    """
    def __init__(self):
        super().__init__()
        human_initial_position=np.array([10.0,2.5,np.pi]).reshape(-1,1)
        self.human_position=human_initial_position
    pass

class Robot(FDM):
    """
    ロボット・ターゲット地点の位置・行動に関する変数を管理
    """
    def __init__(self):
        super().__init__()
        self.target_distance=3.0

        robot_initial_position=np.array([0.5,0.5,0]).reshape(-1,1)
        self.robot_position=robot_initial_position
        self.target_position=self.get_target_position(self.robot_position)

    def get_target_position(self,robot_position):
        x_T=robot_position[0]+self.target_distance*np.cos(robot_position[2])
        y_T=robot_position[1]+self.target_distance*np.sin(robot_position[2])

        return np.array([float(x_T),float(y_T),np.nan]).reshape(-1,1)
    pass


class Visualization(Human,Robot):
    """
    根っこのクラスで定義されたことを図示
    """
    def __init__(self):
        Human.__init__(self)
        Robot.__init__(self)
        pass
    def plot_current_situation(self):
        self.fig = plt.figure()
        self.ax = plt.axes()

        self.plot_character(self.robot_position,"blue")
        self.plot_character(self.human_position,"red")
        self.plot_character(self.target_position,"green")

        self.plot_wall()
        
        plt.axis('scaled')
        self.ax.set_xlim([0,20])
        self.ax.set_ylim([-1,4])
        plt.show()
        pass

    def plot_character(self,position,color):
        if np.isnan(position[2]):
            plt.plot([position[0]-0.25,position[0]+0.25],[position[1]-0.25,position[1]+0.25],c=color)
            plt.plot([position[0]-0.25,position[0]+0.25],[position[1]+0.25,position[1]-0.25],c=color)
            pass
        else: # 有向の場合
            r=0.5
            c = patches.Circle(xy=(position[0], position[1]), radius=r, fc='w', ec=color) # fill color, edge color
            self.ax.add_patch(c)
            plt.plot([position[0],position[0]+r*np.cos(position[2])],[position[1],position[1]+r*np.sin(position[2])])
            # self.ax.annotate('', 
            #                     xy=(position[0]+np.cos(position[2]),position[1]+np.sin(position[2])), 
            #                     xytext=(position[0],position[1]),
            #         arrowprops=dict(shrink=0, width=1, headwidth=8, 
            #                         headlength=10, connectionstyle='arc3',
            #                         facecolor=color, edgecolor=color)
            #    )
    
    def plot_wall(self):
        wall1_y=0.0
        wall2_y=3.0

        plt.plot([0,20],[wall1_y,wall1_y],c="black",lw=0.3)
        plt.plot([0,20],[wall2_y,wall2_y],c="black",lw=0.3)


class Controller(Visualization):
    """
    根っこのクラスで定義されたことを組み合わせて、シミュレーションを実行
    ここで新しい環境情報を定義してはいけない（図示に反映できないから）
    """
    def __init__(self):
        super().__init__()
        pass


    def renew_position(self):
        pass

    def test_walk(self):
        self.robot_position+=np.array([0.5,0,np.pi/12]).reshape(-1,1)
        self.target_position=self.get_target_position(self.robot_position)
        pass
    pass
    

cont=Controller()

for i in range (10):
    cont.test_walk()
    cont.plot_current_situation()