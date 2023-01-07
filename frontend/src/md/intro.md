<script setup lang="ts">
    import Model from '../components/Model.vue'
</script>

# Growing Neural Cellular Automata

---

Christmas is upon us, and that means a short-term migration back to the motherland, and listening to audiobooks in the car.
The book in question for this years treck was [Song of the Cell](https://www.amazon.com/Song-Cell-Exploration-Medicine-Human/dp/1982117354) (a great book about cell biology and the human body).
While listening to the chapter on multicellular organisms, I was reminded of a paper I read a while back, and here we are!

Below is my **reproduction** of the original work of **Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, Michael Levin** in [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). 

In their work, the authors show how to **learn** the update function of a cellullar automata, in such a way, that the cells grow into an arbitrary image, that can regenerate if parts of it are destroyed. Amaze!

See my PyTorch implementation of the code here: [Growing CA's](#link-to-repo), or play around with the some of the automata below!

---

<Suspense>
    <Model :model-file="'0_pool=True_damage=True_epochs=500.onnx'" />
</Suspense>
<Suspense>
    <Model :model-file="'salamander.onnx'" />
</Suspense>

---

### Cellular automata
In it's simplest form, a Cellular Automaton is a discrete model of cells, that reside on a discrete grid, where each point on the grid, corresponds to a cell. Each cell has an internal state, that is iteratively updated, by the application of an update function, that takes as input the current state of the cell and it's immediate neighbours, to produce the next state of each cell.

Depending on the update function of the cell, and the initial conditions, vastly different behaviours of the overall system, i.e. the total grid of cells, will emerge over time. We can think about the update function as the genotype of the cell, and the resulting state of the cell as the phenotype of the cell, or it's particular expression, given it's environment and external stimuli. Following this analogy, we can think of the state of the system, as the phenotype of a multicellular organism.

We can use CA's to model physical phenomena, such as diffusion of particles. In that case, we know the behaviour of each  cell (or particle), but not the resultant emerging behaviour of the entire system, and we can write the rules by hand.
But what if we the converse is the case: We know the final state of the system, but not the rules (the update function) that govern the individual cells? That's the question the authors answer in the original article, and what leads us to the next topic.

### Differentiable programming

A (very) quick primer: Differentiable programming is a programming paradigm where we substitute parts of our program by parameterized differentiable functions. Since the parts are differentiable, we can use gradient descent to find parameters that make our program minimize (or maximize) some differentiable loss function. 

In the case of our CA's, the part of our program that we want to learn, is our update function, and our loss function that we want to minimize, is a function that measures the difference between the total state of our system (our arrangement of cells on our grid), that our program actually produces, and the state that we want it to produce.

### Algorithm

Below, you'll find a (slightly fluffy) description of the algorithm that finds an update function for our cells, that will allow a single cell, to grow into an image of our choosing. The code below is some sort of pseudo-code, with type hints for the tensors (what a world that would be). To see real, working code, please see the [repo](intro.md)

We start with a representation of the state of our system, i.e. of the current state all our cells.
Each cell represents a pixel in a grid (`40x40`), that should eventually grow into an image of our choosing.
We represent the state of each cell, a vector composed of it's visible parts and a hidden state. The authors' propose using a state vector of size `16` where the first four pixels represent a the RGBA pixel representation that we will end up seing, and the remaining `12` entries are the hidden, or internal, state of the cell.

```python
state: Tensor[40, 40, 16] = zeros() # image height x image width x cell state dimension

state[20, 20, 3:] = 1               # Why are we setting the internal state AND the alpha channel of the middle cell to 1?
                                    # read on to find out!
```

The A, in RGBA, or the alpha channel serves a special purpose, apart from it's usual function of designating the opacity of the pixel in the image. The authors also use the alpha channel (`state[cell_x, cell_y, 3]`), to signify the state each cell's life:
* Cells with `state[cell_x, cell_y, 3] == 0` are dead!
* Cells with `0 < state[cell_x, cell_y, 3] < 0.1` are growing..
* And cells with `state[cell_x, cell_y, 3] > 0.1` are fully grown <3


```python
    def whos_alive(state: Tensor[40, 40, 16]) -> Tensor[40, 40, 1]:
        """
            Returns an alive mask,
            where 1.0 represents that the cell at that point on the grid
            is alive and 0.0 represents that it's dead
        """"
        pass
```

Now we're ready to talk about how the denizens of our colony know what is going on around them. The authors represent the cell's perception of the it's surroundings, as the gradient of neighbouring cell states, along the x and y directions of the grid, likening it to the chemical gradients that a biological cell would perceive from its surrounding environment. We convert the state tensor of our grid, to a perception tensor, by convolving the state tensor with two fixed weight sobel convolutions, and concatenating the two outputs with the cells own state, giving us:

```python
sobel_x: Tensor[3, 3] = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

sobel_y = sobel_x.transpose()

def perceive(state: Tensor[40, 40, 16]) -> Tensor[40, 40, 16 * 3]:

    # Below we convolve our state with a horizontal and a vertical sobel filter.
    # What this means, in the horizontal case, is that for each cell state state[cell_x, cell_y, :],
    # we calculate cell_perception_x = state[cell_x - 1, cell_y, :] - state[cell_x + 1, cell_y, :]
    return concat(
        conv2d(state, sobel_x), # The vertical gradient of neighbours
        conv2d(state, sobel_y), # The horizontal gradient 
        state                   # The cells own state
    )
```

To represent the update function, we turn to everybody's favourite differentiable function approximator, the neural network, which will allow the algorithm to find a suitable update function using gradient descent. The NN is a simple feed forward network that takes as input the state of the cell and it's perception of it's surrounding neighbours. To simulate that only some cells update every iteration, we randomly zero out around half of the updates, every update.

```python

network = nn.Sequential(
    nn.Linear(16 * 3, 128), # Horizontal gradient + vertical gradient + own state
    nn.ReLU(),
    nn.Linear(128, 16, bias=False, init=0) # State update delta
                                           # Note that this layer has no bias
                                           # And is initialized with zero weights:
                                           # This makes it default to a NOOP, before we start training it
)

def whos_alive(state: Tensor[40, 40, 16]):
    pass

def update(state: Tensor[40, 40, 16]) -> Tensor[40, 40, 16]:
    """
    Take a system state at time t, and produce the state for time t+1
    """
    perception: Tensor[40, 40, 16 * 3] = perceive(state)
    state_update_delta = network(perception)

    who_grew: Tensor[40, 40, 1] = random_ones(prob=0.5)
    
    updated_state: Tensor[40, 40, 16] = state + state_update_delta * who_grew

    return updated_state * whos_alive(updated_state)

```

Now we have our initial state, a way for our states to perceive their surroundings, and a definition of our update function, and we can implement our training algorithm, to find the parameterization of our update function:

```
TODO: Add training code
```


Describe the training loop

Describe homeostatis training

Describe regen training






We represent our 


Below is the central part of the algorithm:



TODO: Add mathematical/programattic definition of model, i.e. update function + initial state

TODO: Add pseudo code for algorithm

TODO: Write about homeostatis

* Loop until training loss is suffiently low
  * Let **S** be the initial state of our cells, as a tensor grid of cells **H x W x Cell State Dimension**
  * Set **S[H/2, W/2, :] = 1**, simulating one living cell in the initial state, in the center of the image
  * For **k** ~ **uniform(64, 92)** steps
    * 