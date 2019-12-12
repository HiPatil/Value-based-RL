#Custom Environment
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import time
import numpy as np
import random


style.use("ggplot")


class Blob():
    def __init__(self, SIZE = 10):
        self.size = SIZE
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def act(self, choice, diagonal = False):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if diagonal:

            if choice == 0:
                self.move(x=1, y=1)
            elif choice == 1:
                self.move(x=-1, y=-1)
            elif choice == 2:
                self.move(x=-1, y=1)
            elif choice == 3:
                self.move(x=1, y=-1)

        else:
            if choice == 0:
                self.move(x=0, y=1)
            elif choice == 1:
                self.move(x=0, y=-1)
            elif choice == 2:
                self.move(x=-1, y=0)
            elif choice == 3:
                self.move(x=1, y=0)


    def move(self, x=-100, y=-100):

        if x == -100:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if y == -100:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class ENVIRONMENT():



    def __init__(self, num_player=1, num_enemy=1, num_food=1, size = 10, diagonal = False):
        self.size = size
        self.naction = 4
        self.diagonal = diagonal
        self.num_enemy = num_enemy
        self.num_food = num_food
        self.player = Blob(size)
        self.enemy = [Blob() for _ in range(self.num_enemy)]
        self.food = [Blob() for _ in range(self.num_food)]
        self.reward = 0
        self.colors = {1: (255, 0, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}
        self.px,self.py = self.player.x,self.player.y
        self.ex,self.ey = [self.enemy[iter].x for iter in range(self.num_enemy)], [self.enemy[iter].y for iter in range(self.num_enemy)]
        self.fx,self.fy = [self.food[iter].x for iter in range(self.num_food)], [self.food[iter].y for iter in range(self.num_food)]


    def startover(self, newpos=False):

        self.player.x, self.player.y = self.px, self.py
        for iter in range(self.num_enemy):
            self.enemy[iter].x, self.enemy[iter].y = self.ex[iter], self.ey[iter]
        for iter in range(self.num_food):
            self.food[iter].x, self.food[iter].y = self.fx[iter], self.fy[iter]
        if newpos == True:
            self.player = Blob(self.size)
        self.reward = 0

        return (self.player.x, self.player.y), self.reward, False

    def step(self, action):

        self.player.act(action, self.diagonal)
        self.reward = self.calculate_reward()
        return (self.player.x, self.player.y), self.reward

    def calculate_reward(self):

        if self.player.x in [self.enemy[iter].x for iter in range(self.num_enemy)] and self.player.y in [self.enemy[iter].y for iter in range(self.num_enemy)]:
            return -100, True

        if self.player.x in [self.food[iter].x for iter in range(self.num_food)] and self.player.y in [self.food[iter].y for iter in range(self.num_food)]:
            return 100, True

        else:
            return -1, False


    def render(self,renderTime=100):

        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for iter in range(self.num_food):
            env[self.food[iter].x][self.food[iter].y] = self.colors[2]
        for iter in range(self.num_enemy):
            env[self.enemy[iter].x][self.enemy[iter].y] = self.colors[3]
        env[self.player.x][self.player.y] = self.colors[1]
        img = Image.fromarray(env, 'RGB')
        img = img.resize((300, 300))
        cv2.imshow("image", np.array(img))
        cv2.waitKey(renderTime)
        # cv2.destroyAllWindows()

    def sample_action(self):
        return np.random.randint(0, self.naction)
