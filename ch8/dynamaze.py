import gym
from gym import spaces
import pygame
import numpy as np
import time
import pickle

BLOCKS_PATTERN1 = set({
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    
})
SHORTCUT_PATTERN_1 = set({
    
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3,8)
    
})

SHORTCUT_PATTERN_2 = set({

    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    
})

BLOCKS_PATTERN2 = set({

    
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),

})

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":10}
    def __init__(self, render_mode=None, size=(7,9), start=(0,5), goal=(6,8), blocks=BLOCKS_PATTERN1):
        
        
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.blocks = blocks
        
        self.size = size
        
        
        self.window_size = ((1024/7)*9, 1024)
        
        


        self.observation_space = spaces.Tuple((spaces.Box(low=0, high = size[0] - 1, shape=(1,), dtype=int), 
                                               spaces.Box(low=0, high = size[1] - 1, shape=(1,), dtype=int)))


        
        self.action_space = spaces.Discrete(4)
        self.nS = 2* (size[0]*size[1])
        self.nA = 4

        # up, right, down, left (arangement distribution (nth row, nth col) rather than representational (x,y))
        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0,1]),
            2: np.array([-1,0]),
            3: np.array([0,-1]),
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.agent_image = pygame.image.load("r2d2.png") 
        self.agent_image = pygame.transform.scale(self.agent_image, (self.window_size[0]/size[1],self.window_size[1]/size[0]))
        self.agent_rect = self.agent_image.get_rect()
    
        
        self.cheeze_image = pygame.image.load("cheeze.png") 
        self.cheeze_image = pygame.transform.scale(self.cheeze_image, (self.window_size[0]/size[1],self.window_size[1]/size[0]))
        self.cheeze_rect = self.cheeze_image.get_rect()
        
    
    def _get_obs(self):

        return self._agent_location
    
    def _get_info(self):

        return {"distance": np.linalg.norm(self._agent_location - self.goal, ord=1)}
    
    def reset(self, seed = None, options = None):

    
        self._agent_location = self.start
                   
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action):
        direction = self._action_to_direction[action]
        
        
        
        new_position = np.clip(
        self._agent_location + direction, (0, 0) , (self.size[0]-1, self.size[1]-1) )
        
        is_blocked = tuple(new_position) in self.blocks
        
        self._agent_location = new_position if not is_blocked else self._agent_location 
        
        terminated = np.array_equal(self._agent_location, self.goal)
        
#         rewards are sparse (u get 1 when u reach target else u get 0), its called binary sparse rewards lol

        reward = 1 if terminated else 0
    
        observation = self._get_obs()
        
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated,False, info
    
    def render(self):
        if self.render_mode =="rgb_array":
            return self._render_frame()
    
    def _get_pixel_point(self, coord):
        
        n_rows = self.size[0]
        n_cols = self.size[1]
        y = coord[0]
        x = coord[1]
        W = self.window_size[0]/n_cols
        H = self.window_size[1]/n_rows

        d_from_top = (n_rows- 1-y)*H
        d_from_left = (x)*W
        
        return (d_from_left, d_from_top)

  

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        
        # pixel_size = width X Height == (size[1], size[0]) == ncols , nrows
        pix_square_size = ( self.window_size[0] / self.size[1], self.window_size[1] / self.size[0])
        

        # First we draw the goal
        # px=self._get_pixel_point(self.goal)
        # pygame.draw.rect(
        #     canvas,
        #     (0, 70, 255),
        #     pygame.Rect(

        #         px,
        #         (pix_square_size[0], pix_square_size[1]),
        #     ),
        # )
        # Now we draw the agent
        self.agent_rect.x , self.agent_rect.y = self._get_pixel_point(self._agent_location)
        self.cheeze_rect.x,self.cheeze_rect.y = self._get_pixel_point(self.goal)

        # horizontal lines
        for x in range(self.size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size[1] * x),
                (self.window_size[0], pix_square_size[1] * x),
                width=3,
            )
        # vertical lines
        for x in range(self.size[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size[1] * x, 0),
                (pix_square_size[1] * x, self.window_size[1]),
                width=3,
            )
        
        images=[]
        image_rects = []

        for x,y in self.blocks:
            wall_image = pygame.image.load("wall.png") 
            wall_image = pygame.transform.scale(wall_image, (self.window_size[0]/self.size[1],self.window_size[1]/self.size[0]))
            wall_rect= wall_image.get_rect()
            wall_rect.x , wall_rect.y = self._get_pixel_point((x,y))
            canvas.blit(wall_image, wall_rect)

        canvas.blit(self.agent_image, self.agent_rect)
        canvas.blit(self.cheeze_image, self.cheeze_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )




class QControl:
    def __init__(self, nS, nA, terminal=list(range(37,48)), eps=None, initializer=None):
        self.Q = np.random.normal(0.5,0.25, (nS, nA) )

        
        if initializer!=None:
            self.Q = initializer(nS, nA)
        
        
        for i in terminal:
            self.Q[i] = np.array([0, 0, 0, 0])
        
        self.pi = np.random.randint(0, 4, (16, ), dtype=np.int8)
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = eps
        self.last_action = 0
    
    def policy(self, state, eps=None):
        
        if eps == None:
            if self.epsilon != None:
                eps = self.epsilon
            else:
                eps = 0.2
        
        
        if np.random.random() >= eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(0, 4)
        
    def learn(self, s, env, step_size, eps=None):
        if eps == None:
            if self.epsilon != None:
                eps = self.epsilon
            else:
                eps = 0.2
        a = self.policy(s, eps=eps)
        s_, r, done, _, info = env.step(a)
        
        s_ = tuple([*s_["agent"], *s_["target"] ])
        self.Q[s][a] = self.Q[s][a] + step_size*(r + np.max(self.Q[s_]) - self.Q[s][a]   )
        
        self.last_action = a
        
        return s_, r, done,_, info
    
    def plan(self, s, model, step_size,a,  eps=None):
        
                
        r, s_ = model[(s,a)]
        self.Q[s][a] = self.Q[s][a] + step_size*(r + np.max(self.Q[s_]) - self.Q[s][a]   )
        self.last_action = a
        return s_, r, False, {}
    
    def load_q_values(self, file_location):
        f = open(file_location, "rb")
        self.Q = pickle.load(f)

def initializer(nS, nA):
    #     
    indices = np.indices((7,9))
    keys = np.stack(indices, axis=-1).reshape((-1,2))
    Q = {tuple(key): np.random.normal(0.5, 0.25, (nA,)) for key in keys}
    
    return Q

def demo_policy(control, limit=30):
    env.reset()
    done = False
    s = env.reset()
    s = tuple(s[0])
    i = 0
    while not done:
        s_, r, done, prop, _ = env.step(np.argmax(control.Q[s]))
        env.render()
        s_ = tuple(s_)
        s = s_
        i+=1
        if done:
            print("congrats")
        if i >= limit:
            done = True
    

env = MazeEnv(render_mode="human", blocks=SHORTCUT_PATTERN_2)

control = QControl(env.nS, env.nA, eps=0.2, terminal=[], initializer=initializer)
control2 = QControl(env.nS, env.nA, eps=0.2, terminal=[], initializer=initializer)
control.load_q_values("DynaQ_shortcut_plus.pkl")
control2.load_q_values("DynaQ+_shortcut_plus.pkl")

# control.load_q_values("dynaQ_2.pkl")

# for i in range(5):
#     demo_policy(control, limit=30)

env = MazeEnv(render_mode="human", blocks=SHORTCUT_PATTERN_2)

for i in range(5):
    demo_policy(control2, limit=30)
env.close()

