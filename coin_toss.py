# %%
from collections import Counter

import pyro
import torch
from tqdm import tqdm


# %%
def coin_toss(n):
    faces = Counter()
    total = 0
    for _ in tqdm(range(n)):

        # create a sample of Bernoulli distribution for fair (50/50) coin
        # The size of the sample is 1
        face = pyro.sample(
            "face",
            pyro.distributions.Bernoulli(.5)
        ) 


        # convert the Bernoulli distribution into meaning
        # what does 1/0 stands for ?
        face = "head" if face.item() == 1 else "tail"

        # gain 2 if head is tossed, otherwise lose 2
        reward = {"head": 2, "tail": -2}
        total += reward[face]

        # update the faces Counter
        faces[face] += 1 

    return faces, total

def coin_toss_tensor(n):

    # create a sample of Bernoulli distribution for fair (50/50) coin
    # The size of the sample is n
    faces = pyro.sample(
        "faces",
        pyro.distributions.Bernoulli(torch.ones(n) * .5)
    )

    heads = faces.sum()
    tails = n - heads

    total = (heads -  tails) * 2
    result = Counter({"head": heads, "tails": tails}) 
    # return Counter object to summarize the counts of head/tail, and total rewards

    return result, total 


def simulation(n, simulation_func):
    faces, total = simulation_func(n)

    print(f"\nRan {n} simulation{'s' if n >1 else ''}")
    print(f"Total Reward = {total}")
    print(faces)


# %%
for n in [1, 1000, 1000000]:
    simulation(n, coin_toss)


# %%
for n in [1, 1000, 1000000]:
    simulation(n, coin_toss_tensor)

# %%
