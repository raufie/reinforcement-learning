import os
import time


class ScreenState:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.f = lambda: None

    def render(self):
        return f"{self.name}\n{self.value}"

    def set_name(self, name):
        self.name = name
        self.on_change()

    def set_value(self, value):
        self.value = value
        self.on_change()

    def register_on_changed(self, f):
        self.on_change = f


class CMDScreen:
    def __init__(self):
        self.states = []

    def render(self):
        os.system("cls")
        for state in self.states:
            print(state.render())

    def register(self, name, value):
        new_state = ScreenState(name, value)
        new_state.register_on_changed(self.render)
        self.states.append(new_state)
        return new_state

    def unregister(self, state):
        self.states = list(filter(lambda x: id(x) != id(state), self.states))


screen = CMDScreen()
name_object = screen.register("Name", "Rauf")
screen.render()
time.sleep(0.2)

name_object.set_value("BRUH")
