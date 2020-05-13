from diffrascape.env import BadSeeds


def test_construct():
    bad_seeds = BadSeeds()

    bad_seeds_states = bad_seeds.states()
    print(f"### states: {bad_seeds_states}")
    assert bad_seeds_states["shape"][0] == 6

    bad_seeds_actions = bad_seeds.actions()
    print(f"### actions: {bad_seeds_actions}")
    assert bad_seeds_actions["num_values"] == 3
