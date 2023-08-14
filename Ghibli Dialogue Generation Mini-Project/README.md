# Ghibli-Style Dialogue Generation
### Recreate the distinct dialogue style of the world-renowned film studio, Studio Ghibli.

The Japanese animation studio, Studio Ghibli, is well known for its high quality animations and stories. Ghibli films are known for being rich in character, distinct in art style, and enchanting in story. In this project, four films will be used to provide dialogue data: *Castle in the Sky (1986)*, *Howl's Moving Castle (2004)*, *Kiki's Delivery Service (1989)*, and *Spirited Away (2001)*.

This project aims to mimic the studio's distinct dialogue style and generate new dialogue. To achieve this, a Pytorch recursive neural network (specifically an LSTM) is used. Characters from the dialogues of the above four films will be fed into the network, and they will be used to generate new similar dialogue. The neural net will use previous characters to predict upcoming characters, and trains over time to learn patterns in the text. This process allows the net to generate new text based on the most probable next character in the style of these film transcripts.

### Project Overview:
<u>Problem:</u> Generate dialogue text in the style of renowned film studio, Studio Ghibli. 
<br>Topics:
* Pytorch 'long short-term memory network' (LSTM) using character chunks to predict future characters
* Data collection and manipulation, namely reshaping and creating tensors for analysis
* Pytorch LSTM construction and training

### Reproducing this project:
The environment.yml file is provided for this project, so it can be easily reproduced on your machine. Create a new conda environment using the provided file and the libraries/dependencies will be ready to use.

For more information on environment creation, please refer to the Capstone README detailing how to recreate an environment from a .yml file: https://github.com/kevinjin21/SpringboardProjects/tree/main/Capstone/Pharm_Deploy (Refer to part 1., Training the model).

### Additional Notes:
This project uses film transcripts from Studio Ghibli. They are available online, free of charge. This project is for educational purposes only.