<script setup lang="ts">
    import Models from '../components/Models.vue'
</script>

# Reproducing Growing Neural Cellular Automata
##### By Jens Jepsen
---

Christmas is upon us, and that means a short-term migration back to the motherland, and listening to audiobooks in the car. The book in question for this years treck was [Song of the Cell](https://www.amazon.com/Song-Cell-Exploration-Medicine-Human/dp/1982117354) (a great book about cell biology and the human body).
While listening to the chapter on multicellular organisms, I was reminded of a paper I read a while back.

That paper was [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/), and below is **my reproduction** of the original work by **Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin**. 
In their work, the authors show how to **learn** the update function of a cellullar automata, in such a way, that the cells grow into an arbitrary image, that can regenerate if parts of it are destroyed. Amazing!

See my PyTorch implementation of the algorithm and how I deploy the trained automata using ONNX and TypeScript here: [Growing and deploying CA's](#link-to-repo), or play around with the some interactive automata below, to see what all the fuss is about!

---
*Hint: Click the images to interact with them!*

<Models></Models>

---

### Intro
In the paper the authors propose an algorithm for learning the update function of a cellular automata, in such a way that a single progenitor cell, can grow into a predefined target image, after a fixed time of evolution.

The update function learned in this first experiment, will make the system keep growing, and thus if the system is allowed to keep growing, it starts to diverge from the original image.

In their next experiment, they show how to modify the training procedure to learn a function that results in the system converging to a steady state, irrespective of how long we let it evolve.

Next, they show how to make the system robust to sporadic damage to parts of it, making it able to regrow parts of itself, if cells are removed.

### Cellular automata
In it's simplest form, a Cellular Automaton is a discrete model of cells, that reside on a discrete grid, where each point on the grid, corresponds to a cell. Each cell has an internal state, that is iteratively updated, by the application of an update function, that takes as input the current state of the cell and it's immediate neighbours, to produce the next state of each cell.

Depending on the update function of the cell, and the initial conditions, vastly different behaviours of the overall system, i.e. the total grid of cells, will emerge over time. We can think about the update function as the genotype of the cell, and the resulting state of the cell as the phenotype of the cell, or it's particular expression, given it's environment. Following this analogy, we can think of the state of the system, as the phenotype of a multicellular organism.

We can use CA's to model physical phenomena, such as diffusion of particles. In that case, we know the behaviour of each  cell (or particle), but not the resultant emerging behaviour of the entire system, and we can write the rules by hand.
But what if we the converse is the case: We know the final state of the system, but not the rules (the update function) that govern the individual cells? That's the question the authors answer in the original article, and what leads us to the next topic.

To allow the algorithm to learn the update function using gradient descent, or the like, we can use differentiable programming. Differentiable programming is a programming paradigm where we substitute parts of our program by parameterized differentiable functions. Since the parts are differentiable, we can use gradient descent to find parameters that make our program minimize (or maximize) some differentiable loss function. 

In the case of our CA's, the part of our program that we want to learn, is our update function, and our loss function that we want to minimize, is a function that measures the difference between the state that our system has evolved to (our arrangement of cells on our grid), and the state that we want it to produce, our target image.

### Algorithm

Below, you'll find a (slightly fluffy) description of the algorithm that finds an update function for our cells, that will allow a single cell, to grow into an image of our choosing. **To see real, working code**, please see [the repo with my PyTorch implementation](intro.md).

The authors `40x40` images of emojis, and we'll do the same here!

Let's start with the algorithm to evolve our system over time:

```python
def evolve(timesteps):
    """
    Initialize our state grid, where each point in the grid is a vector with the cell state
    We interpret the first four elements of the vector as an RGBA pixel,
    allowing us to evolve our image.
    We initialize the center cell's state vector as [0, 0, 0, 1, ..., 1],
    Note that we let the A of our pixel representation be 1.
    """
    # image height x image width x cell state
    state_grid: Tensor[40, 40, 16] = init_grid()

    for t from range(0, timesteps):
        """
        We start by selecting only the cells that are alive):
        cells with an A value
           = 0 are dead
           < 0.1 are growing
           > 0.1 are fully grown
        Remember that we initialized 1 cell to have an A of 1, in our initial state,
        which means that that's the only cell that starts out being alive.
        """
        alive = mask_alive(state_grid)

        """
        Next, we calculate the perception of each cell
        The authors liken this perception of differences, 
        to a biological cell being aware of chemical gradients in it's environment. 
        """
        perception: Tensor[40, 40, 16 * 3] = concat(
            state_grid,
            horizontal_gradient(state_grid), # cell_state[x-1, y] - cell_state[x+1, y]
            vertical_gradient(state_grid)    # cell_state[x, y-1] - cell_state[x, y+1]
        )

        """
        Now, we calculate the update delta for all our cells.
        The function that calculates our update delta,
        is a neural network, which means we'll be able to train or learn
        the update function. More on that in a minute.
        """
        state_update_delta: Tensor[40, 40, 16] = neural_network(perception)

        """
        Now we're ready to calculate the new state of our system,
        by adding our update delta to the state at t-1.
        To simulate that not all cells grow at the same time,
        we randomly zero out vectors in our update delta with probability 0.5,
        which means that, on average, half of our cells will update at every timestep.
        """
        state_grid: Tensor[40, 40, 16] = state_grid + randomly_zero_out_states(state_update_delta, prob=0.5)
    
    return state_grid

```

Now we have our initial state, a way for our states to perceive their surroundings, a definition of our update function and a way for our cells to evolve, and we can implement our training algorithm, to find the parameterization of our update function:


```python
def train(target_image: Tensor[40, 40, 4], iterations: int):
    """
        This function will train our neural network,
        such that it converges to an update function
        that causes the evolution of our colony of cells
        to converge to the target image
    """
    for _ in range(iterations):
        evolution_steps: int = uniform_int(64, 92)
        evolution_states = evolve(evolution_steps)
        final_state = evolution_states[-1]
        
        """
        Calculate the MSE (mean squared error)
        between the RGBA channels of our final state and our target image,
        after between 64 and 92 update steps
        """
        loss = ((final_state[0:4] - target_image) ** 2).mean()

        """
        Use loss to update NN weights,
        using the Adam optimizer
        (see hyperparameters in the real code in repo)
        """
        update_nn_weights_using_adam(network, loss)
```

Describe homeostatis training

Describe regen training





TODO: Write about homeostatis