class StableMapping:
    def __init__(self, colors):
        self.colors = colors
        self.available_colors = colors.copy()
        self.mapping = {}
        self.last_assigned = {}

    def map(self, ids):
        result = []
        for id in ids:
            if id in self.mapping:
                color = self.mapping[id]
            else:
                if self.available_colors:
                    color = self.available_colors.pop(0)
                else:
                    color = min(self.last_assigned, key=self.last_assigned.get)
                self.mapping[id] = color
            self.last_assigned[color] = len(result)
            result.append(color)
        for id in list(self.mapping.keys()):
            if id not in ids:
                color = self.mapping.pop(id)
                if color not in self.mapping.values():
                    self.available_colors.insert(1, color)
        return result

    def __getitem__(self, id):
        return self.mapping[id]

    def reset(self):
        self.available_colors = self.colors.copy()
        self.mapping = {}
        self.last_assigned = {}


def test_stable_mapping():
    mapping = StableMapping(["red", "blue", "green"])

    assert mapping.map([1, 2]) == ["red", "blue"]
    assert mapping.map([2]) == ["blue"]
    assert mapping.map([2, 1]) == ["blue", "green"]
    assert mapping.map([2, 1, 2]) == ["blue", "green", "blue"]
    assert mapping.map([2, 1, 3, 4]) == ["blue", "green", "red", "blue"]


if __name__ == "__main__":
    test_stable_mapping()
    print("StableMapping tests passed.")
