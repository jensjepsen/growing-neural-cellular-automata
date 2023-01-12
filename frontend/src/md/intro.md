<script setup lang="ts">
    import Models from '../components/Models.vue'
</script>

# Reproducing Growing Neural Cellular Automata
##### By Jens Jepsen
---

Below is **my reproduction** of the original work by **Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin**, in [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). 
In their work, the authors show how to **learn** the update function of a cellullar automata, in such a way, that the cells grow into an arbitrary image, that can regenerate if parts of it are destroyed. Amazing!

See my PyTorch implementation of the algorithm and how I deploy the trained automata using ONNX and TypeScript here: [Growing and deploying CA's](https://github.com/jensjepsen/growing-neural-cellular-automata), or play around with the interactive automata below, to see what all the fuss is about!

---
*Hint: Click the images to interact with them!*

<Models></Models>

---

### Intro

Christmas is upon us, and that means a short-term migration back to the motherland, and listening to audiobooks in the car. The book in question for this years treck was [Song of the Cell](https://www.amazon.com/Song-Cell-Exploration-Medicine-Human/dp/1982117354) (a great book about cell biology and the human body).
While listening to the chapter on multicellular organisms, I was reminded of the article, that I read a while back.

In the paper the authors propose an algorithm for learning the update function of a cellular automata, in the form of a neural network, in such a way that a single progenitor cell, can grow into a predefined target image, after a fixed time of evolution.

The basic algorithm proceeds by iteratively doing the following:
* Evolve the grid of cells using the current neural network update function
* Calculate the difference between the visible part of evolved state and a target image
* Adjust the weights of the neural network, using gradient descent, in such a way way that the next evolution of will be closer
* Repeat!

The authors perform four experiments:

* Learn an update function that will make the system grow into the target image, after k timesteps, where k is a randomly sampled integer drawn from a uniform distribution. If this experiment is allowed to keep growing, it starts to diverge from the original image, as you can see above on the automata labelled **growing**.

* Modify the training procedure to learn a function that results in the system converging to a steady state, irrespective of how long we let it evolve. See **stable**, above.

* Further adjust the training procedure, to make the system robust to sporadic damage and enabling it to regrow parts of itself, if cells are removed. This is the coolest one - see it above under **regenerating**.

* TODO: Rotation

### Cellular automata
In it's simplest form, a Cellular Automaton is a discrete model of cells, that reside on a discrete grid, where each point on the grid corresponds to a cell. Each cell has an internal state, that is iteratively updated, by the application of an update function that takes as input the current state of the cell and it's immediate neighbours, to produce the next state of each cell.

Depending on the update function of the cell, and the initial conditions, vastly different behaviours of the overall system, i.e. the total grid of cells, will emerge over time. We can think about the update function as the genotype of the cell, and the resulting state of the cell as the phenotype of the cell, or it's particular expression, given it's environment. Following this analogy, we can think of the state of the system, as the phenotype of a multicellular organism.

We can use CA's to model physical phenomena, such as diffusion of particles. In that case, we know the behaviour of each  cell (or particle), but not the resultant emergent behaviour of the entire system, and we can write the rules by hand.
But what if we the converse is the case: We know the final state of the system, but not the rules (the update function) that govern the individual cells? That's the question the authors answer in the original article, and what leads us to the next topic.

To allow the algorithm to learn the update function using gradient descent, or the like, we can use differentiable programming. Differentiable programming is a programming paradigm where we substitute parts of our program by parameterized differentiable functions. Since the parts are differentiable, we can use gradient descent to find parameters that make our program minimize (or maximize) some differentiable loss function. 

In the case of our CA's, the part of our program that we want to learn, is our update function, and our loss function that we want to minimize, is a function that measures the difference between the state that our system has evolved to (our arrangement of cells on our grid), and the state that we want it to produce, our target image.

### Algorithm

Below, you'll find a (slightly fluffy) description of the algorithm that finds an update function for our cells, that will allow a single cell, to grow into an image of our choosing. **To see real, working code**, please see [the repo with my PyTorch implementation](https://github.com/jensjepsen/growing-neural-cellular-automata).

The authors use `40x40` images of emojis, and we'll do the same here!

Let's start with the algorithm to evolve our system over time. This procedure remains the same over all the experiments.
Please note, that in practice, we'll be evolving a batch of cell grids in every iteration, instead of just one, to stabilize the training produce. In the pseudo-code we mostly pretend that we'll just be evolving a single grid at a time.

```python
def evolve(timesteps: int, initial_state: Tensor[40, 40, 16], neural_network: NeuralNetwork):
    """
    This function takes an initial state,
    and a neural network that represents our update function,
    and evolves the state of cells on the grid for a number of timesteps.
    """
    

    """
    Our initial state is a 40x40 grid,
    where each point on the grid is a 16 dimensional vector that represents the cell state.
    The first 4 elements of the vector are the components of an RGBA pixel, 
    and the last 12 are the cells internal state.
    """
    state_grid: Tensor[40, 40, 16] = intial_state.copy()
    for t from range(0, timesteps):
        """
        The fourth element of a cells state,
        or the A channel of the RGBA pixel, has a special significance,
        apart from denoting the opacity of the pixel.
        We also use it to denote the lifecycle of a cell.
        This leads to the following rule for a cell's lifecycle:
           state_grid[:, :, 3] = 0, dead
           0 < state_grid[:, :, 3] = < 0.1, growing
           state_grid[:, :, 3] > 0.1, mature

        We first select the cells who are mature themselves,
        or that have neighbours that are:
        """
        alive = select_where(state_grid, max_pool_2d(state_grid, kernel_size=(3, 3)) > 0.1)

        """
        Next, we calculate the perception of each cell,
        as the gradient along the vertical and horizontal axis of the grid,
        concatenated with the cell's own internal state.
        The authors liken this perception of gradients, or differences, 
        to a biological cell being aware of chemical gradients in it's environment. 
        """
        perception: Tensor[40, 40, 16 * 3] = concat(
            state_grid,
            horizontal_gradient(alive), # cell_state[x-1, y] - cell_state[x+1, y]
            vertical_gradient(alive)    # cell_state[x, y-1] - cell_state[x, y+1]
        )

        """
        Now, we calculate the update delta for all our cells.
        The function that calculates our update delta,
        is a neural network, which means we'll be able to train, or learn,
        the update function, adjusting it's parameters to minimize a loss.
        More on that in a minute.
        """
        state_update_delta: Tensor[40, 40, 16] = neural_network(perception)

        """
        Finally, we calculate the new state of our system,
        by adding our update delta to the state at t-1.
        To simulate that not all cells grow at the same time,
        we randomly zero out cell state vectors in our update delta with probability 0.5,
        which means that, on average, half of our cells will update at every timestep.
        """
        state_grid: Tensor[40, 40, 16] = state_grid + randomly_zero_out_states(state_update_delta, prob=0.5)
    
    return state_grid

```

Now we move to the training procedure, where we train the neural neural network to learn the update function the we want:

```python
def train(target_image: Tensor[40, 40, 4], iterations: int, network: NeuralNetwork):
    """
    This function will train our neural network,
    such that it converges to an update function
    that causes the evolution of our colony of cells
    to converge to the target image.
    """

    """
    We start by initializing our grid
    to only have a single living cell, in the center.
    """
    # image height x image width x cell state
    state_grid: Tensor[40, 40, 16] = init_grid(only_center_cell_alive=True)


    for _ in range(iterations):
        """
        In each iteration of this loop,
        we evolve the state for k ~ uniform(64, 92) timesteps,
        and compare the visible part of the final state
        to the image we want it to produce.
        """
        
        evolution_steps: int = uniform_int(64, 92)

        final_state = evolve(evolution_steps, neural_network)
        
        visible_state = final_state[:, :, 0:4]

        """
        Calculate the mean squared error
        between the RGBA channels of our final state and our target image,
        after between 64 and 92 update steps
        """
        loss = ((visible_state - target_image) ** 2).mean() / 2

        """
        Update NN weights,
        in the direction that minimizes our the difference between the
        evolved state, and our target image.
        """
        update_nn_weights(network, loss)
```

The training procedure described by the pseudo-code above, implements the first experiment, and forms the basis of the basis of the next experiments.
As you can see above under **growing** this system starts devolving, for most of the images, if we let it run long enough. For some of the images, such as the spiderweb, it continues building of the previous pattern, which looks pretty cool. But for some of the others, it quickly turns scary! 

To learn an update function that converges to a steady-state, we can modify the training procedure slightly. Instead of always letting the system evolve from the same fixed state, we maintain a pool of previously observed states, and sample an initial state from that pool, at each iteration, to use as our initial condition.

*Note: In practice, when doing batch training, we sample k previous states, and sort them in descending order of loss. The state sampled state with the highest loss is then replaced with the single cell intitial state from above. The authors note that this is needed to stabilize training. But let's ignore that fact here, to make my life easier.*

```python
sample_pool = PoolOfTensors(capacity=1024)

def init_grid_from_pool(batch_size):
    return sample_pool.sample()



def train(target_image: Tensor[40, 40, 4], iterations: int, network: NeuralNetwork):
    """
        We update the training function to sample from the pool,
        and store states in the pool for later use.
    """
    state_grid: Tensor[40, 40, 16] = init_grid_from_pool()

    for _ in range(iterations):
        evolution_steps: int = uniform_int(64, 92)
        final_state = evolve(evolution_steps, neural_network)

        """
        We add the final state to the sample pool here.
        Everything else remains unchanged.
        """
        sample_pool.add(final_state)

        visible_state = final_state[:, :, 0:4]
        loss = ((visible_state - target_image) ** 2).mean() / 2
        update_nn_weights(network, loss)
```

TODO: Describe regen training