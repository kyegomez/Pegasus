[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)



<div align="center">


</div>


# PegasusX: The Future of Multimodal Embeddings 🦄 🦄 

![Pegasus Banner](stable-diffusion-xl.jpeg)

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Pegasus)](https://github.com/kyegomez/Pegasus/issues)
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/Pegasus)](https://github.com/kyegomez/Pegasus/network)
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/Pegasus)](https://github.com/kyegomez/Pegasus/stargazers)
[![GitHub license](https://img.shields.io/github/license/kyegomez/Pegasus)](https://github.com/kyegomez/Pegasus/blob/main/LICENSE)
[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/Pegasus?style=social)](https://star-history.com/#kyegomez/Pegasus)
[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/Pegasus)](https://libraries.io/github/kyegomez/Pegasus)
[![Downloads](https://static.pepy.tech/badge/pegasusx/month)](https://pepy.tech/project/pegasusx)

### Share on Social Media

[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20GitHub%3A%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus)
[![Facebook](https://img.shields.io/badge/Share-Facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus&title=&summary=&source=)
[![Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus&title=PegasusX%20-%20The%20Future%20of%20Multimodal%20Embeddings)
[![Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus&t=PegasusX%20-%20The%20Future%20of%20Multimodal%20Embeddings)
[![Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=PegasusX%20-%20The%20Future%20of%20Multimodal%20Embeddings)
[![WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20PegasusX%20-%20The%20Future%20of%20Multimodal%20Embeddings%20%23Pegasus%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FPegasus)

</div>


Welcome to PegasusX, the latest and most advanced package for creating high-quality embeddings from multimodal data. We're pushing the boundaries of what's possible with machine learning, enabling tasks and applications that were once mere visions of the future.

In essence, PegasusX is designed to transform the way we look at data. Our aim is to make it easier for anyone, regardless of their domain or discipline, to generate task-specific, high-quality embeddings from any type of data, be it text, image, video, audio, or even more complex data types..

# Documentation
* [Click here for documentation](DOCs/DOCUMENTATION.md)

## Installation

Sure, here is the modified section including installation instructions using `git clone`:

# Git Clone Installation

There are 2 methods of installation. Currently, we're experiencing some path errors with pip installation. For a smooth installation, we recommend using git clone:

```bash
git clone https://github.com/kyegomez/Pegasus.git
cd Pegasus
pip install -r requirements.txt
```

To validate your installation, you can run the provided example:

```bash
python3 example.py
```

## Usage

```python
from pegasus import Pegasus

# For video and audio modalities, you can initialize the Pegasus class with "Pegasus('vision')" or "Pegasus('audio')" respectively, then pass in the file path of the vision or audio data
pegasus = Pegasus("text", multi_process=False, n_processes=4)

text_data = [
    'This is a query about artificial intelligence',
    'Another query about machine learning',
    'Yet another query about deep learning',
    'And one more about natural language processing'
]

embeddings = pegasus.embed_data(text_data)

print(embeddings)
```

## Pip Installation
Please help us with this file path errors, they are very annoying.

```bash
pip install pegasusx
```

## Pip Usage

```python
from pegasus import Pegasus

# for video, audio do "Pegasus('vision'), Pegasus("audio") respectively then pass in the file path of the vision or audio data
pegasus = Pegasus("text", multi_process=False, n_processes=4)

text_data = ['This is a query about artificial intelligence',
             'Another query about machine learning',
             'Yet another query about deep learning',
             'And one more about natural language processing']

embeddings = pegasus.embed_data(text_data)

print(embeddings)
```

## Features

PegasusX is not just another run-of-the-mill machine learning package. We've painstakingly crafted this package, ensuring it includes features that set it apart:

1. **Multimodal Data Understanding:** From text to images, audio, and more, PegasusX is designed to handle and understand a wide array of data types.

2. **Personalized for Any Task:** PegasusX adapts to your specific task, generating high-quality, task-specific embeddings for a wide variety of applications.

3. **Scalability & Performance:** PegasusX has been optimized for efficiency and can scale according to the demands of your tasks, ensuring seamless operation even with large amounts of data.

4. **Open Source:** We believe in the power of community and collaboration. PegasusX is an open-source project, welcoming contributions and improvements from the global developer community.

## Contributing to PegasusX

We are thrilled to invite you to be a part of the PegasusX project. This is not just an open source project but a community initiative, and we value your expertise and creativity. To show our appreciation, we have instituted a unique rewards system that directly compensates contributors from the revenue generated by the PegasusX API.

### Why Contribute

Contributing to PegasusX not only enhances your skills and profile but also comes with financial rewards. When you contribute code, documentation, or any form of improvement to the PegasusX project, you are adding value. As such, we believe it's only fair that you share in the rewards.

### Rewards Program

Here's how the PegasusX Rewards Program works:

1. **Submit a Pull Request:** This can be a code enhancement, bug fix, documentation update, new feature, or any improvement to the project.

2. **Review and Approval:** Our team will review your contribution. If it gets approved and merged, you become eligible for the rewards program.

3. **Revenue Share:** Once your pull request is merged, you will receive a percentage of the revenue generated by the PegasusX API. The percentage will be determined based on the significance and impact of your contribution. 

## Becoming a Paid API

As part of our growth strategy, we will be deploying PegasusX as a Paid API. The revenue generated from this API will not only sustain and further the project, but also fund the rewards program.

### How to Start Contributing

If you're ready to become a part of PegasusX and contribute to the future of multimodal embeddings, here's what you need to do:

1. Fork the repository.

2. Make your improvements or additions in your forked repository.

3. Submit a pull request detailing the changes you've made.

4. Our team will review your submission. If it's approved, it will be merged into the main repository, and you will become part of the PegasusX Rewards Program.

## Roadmap

PegasusX is a constant work in progress, and we're always striving for better. Our roadmap provides a snapshot of where we're heading. 

* **Reconfiguring the ImageBind Model:** To improve our handling of diverse data, we are reconfiguring the ImageBind model to utilize Flash Attention. This shift will allow us to manage longer context lengths and handle more complex inputs effectively.

* **Pretraining with Diverse Datasets:** Quality embeddings require quality training. To ensure our model is versatile and robust, we're pretraining PegasusX using the same datasets that ImageBind has been trained on. This step ensures our model inherits the benefits of proven training methodologies while also incorporating our enhancements.

* **Benchmarking:** It's important to know where we stand. After we've reconfigured and pretrained our model, we will conduct comprehensive benchmarking tests. This process will highlight any areas of strength or potential improvement, allowing us to further refine our model.

* **Finetuning on Long Samples:** We believe that PegasusX can handle more than short snippets of data. To prove this, we'll finetune our model using long data samples, pushing the boundaries of what's possible with embedding models.

* **Continued Innovation:** Our roadmap doesn't stop with finetuning. As we move forward, we're excited to explore new methodologies and techniques to enhance PegasusX. 

* **Advanced Training Techniques:** We'll look into more sophisticated methods to make the training process faster and more efficient.

* **Expanding Modality Types:** We aim to support more types of modalities, ensuring that PegasusX is truly a universal tool for multi-modal data.

* **Integration with More Frameworks:** We want PegasusX to be accessible and easy to use with popular machine learning and data processing frameworks.

* **Optimizing for Real-Time Processing:** We're focused on making PegasusX capable of generating embeddings in real-time, a critical feature for many applications.

* **Community Driven Enhancements:** We're excited to see what the community suggests and contributes - the possibilities are endless!

* **Production-Level API Deployment:** PegasusX will enter Agora's paid API line up so you can effortlessly make API requests and receive your embeddings no complicated setup necessary

* **Making it Extremely Fast Through Quantization:** By utilizing quantization techniques, we aim to significantly increase the speed and efficiency of the PegasusX model.

* **Parallelization, Asynchrony, and Other Optimizations:** To ensure seamless operation even with large amounts of data, we're planning to implement parallelization, asynchronous operations, and other optimizations in the model.

* Remake in Jax using dynamic sparse flash attention

Thank you for considering contributing to PegasusX. Your expertise and commitment to this project are what make it thrive. Let's build the future of multimodal embeddings together.


# Demo


---

# Demos

## Swarm Video Demo {Click for more}

[![Watch the swarm video](https://img.youtube.com/vi/V4JE5YBlkpU/maxresdefault.jpg)](https://youtu.be/V4JE5YBlkpU)

----
