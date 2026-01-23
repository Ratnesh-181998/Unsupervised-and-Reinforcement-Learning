# 𝗟𝗲𝘁’𝘀 𝘀𝘁𝗮𝗿𝘁 𝘄𝗶𝘁𝗵 𝗨𝗻𝘀𝘂𝗽𝗲𝗿𝘃𝗶𝘀𝗲𝗱 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗮𝗻𝗱 𝘁𝗵𝗲𝗻 𝗺𝗼𝘃𝗲 𝗼𝗻 𝘁𝗼 𝗶𝗻-𝗱𝗲𝗽𝘁𝗵 𝗗𝗲𝗲𝗽 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗰𝗼𝗻𝗰𝗲𝗽𝘁𝘀

---
- Unsupervised learning is a class of machine learning techniques that aims to discover hidden patterns, structure, or representations in data 𝘄𝗶𝘁𝗵𝗼𝘂𝘁 𝘁𝗵𝗲 𝘂𝘀𝗲 𝗼𝗳 𝗹𝗮𝗯𝗲𝗹𝗲𝗱 𝗼𝘂𝘁𝗽𝘂𝘁𝘀. Unlike supervised learning, where models learn a mapping from inputs to known targets, unsupervised learning operates only on the 𝗶𝗻𝗽𝘂𝘁 𝘀𝗽𝗮𝗰𝗲 𝗮𝗻𝗱 𝗿𝗲𝗹𝗶𝗲𝘀 𝗼𝗻 𝗶𝗻𝘁𝗿𝗶𝗻𝘀𝗶𝗰 𝗽𝗿𝗼𝗽𝗲𝗿𝘁𝗶𝗲𝘀 𝗼𝗳 𝘁𝗵𝗲 𝗱𝗮𝘁𝗮. It is essential for exploratory data analysis, feature learning, dimensionality reduction, clustering, and representation learning in modern AI systems.
Let the dataset be 𝗫 = {𝘅₁, 𝘅₂, …, 𝘅ₙ}, where each xᵢ ∈ Rᵈ. The objective of unsupervised learning is to model the underlying distribution p(x) or to identify meaningful structure within X. One major category is 𝗰𝗹𝘂𝘀𝘁𝗲𝗿𝗶𝗻𝗴, where the goal is to 𝗽𝗮𝗿𝘁𝗶𝘁𝗶𝗼𝗻 𝗱𝗮𝘁𝗮 𝗶𝗻𝘁𝗼 𝗴𝗿𝗼𝘂𝗽𝘀 𝘁𝗵𝗮𝘁 𝗺𝗮𝘅𝗶𝗺𝗶𝘇𝗲 𝘄𝗶𝘁𝗵𝗶𝗻-𝗴𝗿𝗼𝘂𝗽 𝘀𝗶𝗺𝗶𝗹𝗮𝗿𝗶𝘁𝘆 𝗮𝗻𝗱 𝗺𝗶𝗻𝗶𝗺𝗶𝘇𝗲 𝗯𝗲𝘁𝘄𝗲𝗲𝗻-𝗴𝗿𝗼𝘂𝗽 𝘀𝗶𝗺𝗶𝗹𝗮𝗿𝗶𝘁𝘆. In K-means clustering, this is formulated as minimizing the objective: 𝗝 = Σₖ Σ_{𝘅ᵢ∈𝗖ₖ} ||𝘅ᵢ − μₖ||²

- where μₖ is the 𝗰𝗲𝗻𝘁𝗿𝗼𝗶𝗱 𝗼𝗳 𝗰𝗹𝘂𝘀𝘁𝗲𝗿 𝗖ₖ. This objective encourages compact and well-separated clusters.
- Another important class of unsupervised methods is 𝗱𝗶𝗺𝗲𝗻𝘀𝗶𝗼𝗻𝗮𝗹𝗶𝘁𝘆 𝗿𝗲𝗱𝘂𝗰𝘁𝗶𝗼𝗻, which seeks a low-dimensional representation z ∈ Rᵏ, k < d, that preserves important information. - In 𝗣𝗿𝗶𝗻𝗰𝗶𝗽𝗮𝗹 𝗖𝗼𝗺𝗽𝗼𝗻𝗲𝗻𝘁 𝗔𝗻𝗮𝗹𝘆𝘀𝗶𝘀, the transformation is linear and defined as z = Wᵀx, where W consists of eigenvectors of the covariance matrix Σ = (1/𝗻)𝗫ᵀ𝗫. 
- The objective is to 𝗺𝗮𝘅𝗶𝗺𝗶𝘇𝗲 𝘃𝗮𝗿𝗶𝗮𝗻𝗰𝗲: 𝗺𝗮𝘅 𝗩𝗮𝗿(𝗪ᵀ𝗫).
- Density estimation is another core unsupervised task, where the goal is to approximate p(x). 𝗚𝗮𝘂𝘀𝘀𝗶𝗮𝗻 𝗠𝗶𝘅𝘁𝘂𝗿𝗲 𝗠𝗼𝗱𝗲𝗹𝘀 represent the distribution as a weighted sum of Gaussians:𝗽(𝘅) = Σ πₖ 𝗡(𝘅 | μₖ, Σₖ)
- Parameters are learned using the 𝗘𝘅𝗽𝗲𝗰𝘁𝗮𝘁𝗶𝗼𝗻–𝗠𝗮𝘅𝗶𝗺𝗶𝘇𝗮𝘁𝗶𝗼𝗻 𝗮𝗹𝗴𝗼𝗿𝗶𝘁𝗵𝗺, which alternates between computing responsibilities and maximizing likelihood.
- In modern generative models such as 𝗮𝘂𝘁𝗼𝗲𝗻𝗰𝗼𝗱𝗲𝗿𝘀, unsupervised learning minimizes reconstruction loss: 𝗟 = ||𝘅 − 𝗳(𝗴(𝘅))||²
where g is an encoder and f is a decoder. Unsupervised learning thus enables machines to discover structure, compress information, and learn representations, forming the foundation for clustering, anomaly detection, and generative AI systems.

---

# The role of Reinforcement Learning (RL)

- [Reinforcement Learning ](https://github.com/Ratnesh-181998/AI-Engineer/blob/main/Reinforcement%20Learning%20(RL)%20is%20a%20type%20of%20machine%20Learning.pdf)
  
<img width="346" height="326" alt="image" src="https://github.com/user-attachments/assets/8acbd862-2c83-408e-b59b-0d01755974aa" />
<img width="618" height="845" alt="image" src="https://github.com/user-attachments/assets/5daa222e-ddcb-41a5-8da9-551efe9c0f01" />
<img width="505" height="563" alt="image" src="https://github.com/user-attachments/assets/bb2f1505-ab2b-4ed2-be39-f4a2a602c709" />
<img width="493" height="430" alt="image" src="https://github.com/user-attachments/assets/dd1ab5e8-b849-46b1-bff7-7099ce5a7f8a" />
<img width="493" height="506" alt="image" src="https://github.com/user-attachments/assets/44fc97cf-1da2-42d6-9d94-94e5434cfe40" />
<img width="488" height="682" alt="image" src="https://github.com/user-attachments/assets/184b4c1c-0b4f-45a5-8947-2aa459c9cb69" />

---


