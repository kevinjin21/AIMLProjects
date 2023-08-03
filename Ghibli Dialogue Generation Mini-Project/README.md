# Ghibli-Style Dialogue Generation
### Recreate the distinct dialogue style of the world-renowned film studio, Studio Ghibli.

The Japanese animation studio, Studio Ghibli, is well known for its high quality animations and stories. Ghibli films are known for being rich in character, distinct in art style, and enchanting in story. In this project, four films will be used to provide dialogue data: *Castle in the Sky (1986)*, *Howl's Moving Castle (2004)*, *Kiki's Delivery Service (1989)*, and *Spirited Away (2001)*.

This project aims to mimic the studio's distinct dialogue style and generate new dialogue. To achieve this, a Pytorch recursive neural network (specifically an LSTM) is used. Characters from the dialogues of the above four films will be fed into the network, and they will be used to generate new similar dialogue.

### Project Overview:
<u>Problem:</u> Generate dialogue text in the style of renowned film studio, Studio Ghibli. 
<br>Topics:
* Pytorch 'long short-term memory network' (LSTM) using character chunks to predict future characters
* Data collection and manipulation, namely reshaping and creating tensors for analysis
* Pytorch LSTM construction and training