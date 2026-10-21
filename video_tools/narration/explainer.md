# NEAT Bipedal Walker explainer

## title

This is a neural network learning to walk. Nobody trained it with gradients. It was evolved, weights and shape together, by an algorithm called NEAT.

## task

This is Bipedal Walker. A box-shaped body sits on two legs, and each leg has a hip and a knee. The ground is bumpy, and it is different every time.

## task.2

At every step, the walker sets four motor torques, one for each joint, each somewhere between minus one and plus one.

## task.3

The goal is to walk to the far end without falling, within one thousand six hundred steps. That is about thirty two seconds of simulated time.

## senses

To decide, the walker senses twenty four numbers. Four describe the body: how it is tilted, how fast it spins, and how fast it moves.

## senses.2

Ten describe the legs: the angle and speed of every joint, and whether each foot is touching the ground.

## senses.3

And ten come from a lidar, a fan of rays that measures how far away the ground is ahead. All of them are scaled to a fixed range, so a trained walker is one small file.

## reward

The score comes from the environment itself. Moving forward earns points, about three hundred for walking the whole course.

## reward.2

Every bit of motor effort costs a little, so flailing lowers the score.

## reward.3

And falling is expensive. If the body touches the ground, the episode ends with minus one hundred. The task counts as solved at an average of three hundred, over one hundred random terrains.

## network

The pilot is a small neural network. The twenty four sensor values come in on the left and flow through hidden nodes to four outputs on the right.

## network.2

Here it is while walking. Blue means a positive signal, orange a negative one. Thicker lines are stronger connections.

## network.3

The four outputs on the right are the joint torques, sent to the motors at every step.

## hidden

This project had one hard rule: the network must have at least one hidden layer. So every network starts as twenty four inputs, eight hidden nodes and four outputs, with no shortcut from an input straight to a motor.

## hidden.2

Evolution could still remove nodes, so the code checks. It counts the hidden nodes on a real path from input to output. A network with none gets its fitness cut, and it can never be saved as a champion.

## neat

NEAT starts with a population of two hundred random networks. Each one is described by a genome: a list of its nodes and its weighted connections.

## neat.2

Children differ from their parents by mutation. Most mutations just change a weight. Some add a new connection. And some add a whole new node, by splitting a connection in two. That is how the networks grow.

## neat.3

Each generation, the best fifth survive and have children. A child is a copy, or a mix of two parents, plus mutations. The two very best pass on unchanged.

## neat.4

Similar networks are grouped into species, and they mostly compete within their own species. That gives a new structure a few generations to prove itself.

## fitness

How good is a network? Every network walks the same three random terrains, and its fitness is the average score. The terrains change every generation, so nobody can memorise them. Later in the run, that went up to 5 terrains.

## fitness.2

To save time, an episode stops early if the walker makes no real progress for one hundred and fifty steps. A network that just stands there does not waste one thousand six hundred steps of compute.

## fitness.3

But this creates a trap. Standing still scores about minus nine. A brave attempt that falls scores minus one hundred. So at first, standing still wins.

## hof

The best score in each generation is noisy. Three terrains can be lucky, or unlucky.

## hof.2

So every ten generations, the top five networks walk ten terrains they have never seen, with no early stop. The best average becomes the champion. That is the blue line.

## curve

Here is the whole run. For the first 70 generations or so, the best networks simply stood still. The trap held.

## curve.2

Then, around generation 69, a network found a slow shuffle that moved forward without falling. Its score jumped above zero, and the population followed.

## curve.3

The champion passed 100 at generation 99. The scores had levelled off, so from generation 430 each network got 5 terrains instead of three, and the population grew to 300. After 674 generations, its average on ten unseen terrains is 200.

## stages

Here are the champions side by side, each on a typical terrain: the middle one of its ten test terrains. From a random network, to a shuffle, to a real walk.

## stages.2

And this is the final champion.

## growth

While the scores rose, the networks grew. The violet line counts hidden nodes in use, and the amber line counts layers from input to output.

## growth.2

The champion has 6 hidden nodes in 5 hidden layers, and only 37 connections. It started with eight hidden nodes in one layer and 224 connections. Evolution threw most of them away, and stacked what was left into a deeper chain.

## species

This is the number of species. A setting called the compatibility threshold decides how different two networks must be to count as separate species, and the code tunes it to aim for about ten.

## species.2

In the first hour, that tuning ran away. neat python never merges species, so the count barely fell, and the threshold kept climbing, to thirty eight. It is now kept within limits, and the run continued from its checkpoint.

## results

The real test is one hundred terrains the champion has never seen. Its average is 179. Half of its runs score between 191 and 220.

## results.2

But it falls in 17 of the one hundred, and each fall costs a lot. That is what pulls the average down, and it is the main thing left to fix.

## results.3

For comparison, gradient based methods on the same laptop reached about two hundred and seventy with PPO, and just over three hundred with TD3. NEAT is 93 points behind PPO, without computing a single gradient.

## laptop

All of this ran on a four core laptop with no graphics card, in 8 hours of training. Every network is evaluated on its own, so evolution spreads easily across the cores.

## outro

No gradients. No GPU. Just a population of networks, mutating, competing, and growing. The code, the clips and the data are on the project page.

