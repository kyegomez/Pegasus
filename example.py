# pip install pegasusx
from pegasus.main import Pegasus

# # initialize with text modality
# pegasus_text = Pegasus(modality="text")
# text_data = ['This is a query about artificial intelligence']
# embeddings_text = pegasus_text.embed_text(text_data)

# # initialize with audio modality
# pegasus_audio = Pegasus(modality="audio")
# audio_data = [...]  # Your audio data here => audio file apth
# embeddings_audio = pegasus_audio.embed_audio(audio_data)


# text
pegasus = Pegasus(modality="text")  # audio or vision

text_data = [
    "This is a query about artificial intelligence",
    "Another query about machine learning",
    "Yet another query about deep learning",
    "And one more about natural language processing",
]
embeddings = pegasus.embed_data(text_data)
print(embeddings)


# audio
pegasus = Pegasus(modality="audio")

audio_data = "./audio.mp3"  # file path of your mpt3

embeddings = pegasus.embed_data(audio_data)
print(embeddings)

# vision
pegasus = Pegasus(modality="vision")

vision_data = "stable-diffusion-xl.jpeg"

embedding = pegasus.embed_data(vision_data)

print(embedding)
