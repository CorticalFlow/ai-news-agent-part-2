# ai-news-agent-part-2

## Instalation  q

### Create a conda environment 
To encapsulate the dependencies we will use a conda environment([checkout Conda Docs for more information](https://www.anaconda.com/docs/getting-started/miniconda/main)).

```bash
conda create -n ai-news-agent python=3.10
conda activate ai-news-agent
```

### Checkout the code (on client and server)

```bash
git clone https://github.com/corticalflow/ai-news-agent-part-2.git
cd ai-news-agent
```

Install the dependencies    
```bash

# client
pip install -r requirements_client.txt

# server
pip install -r requirements_server.txt
pip install tensorflow[and-cuda]
pip install tf-keras


```

create data folder for processed frames   
```bash
mkdir -p ./data
```


# Read our full Tutorial Article on https://corticalflow.com/blog/Implementing-AI-Powered-News-Agent-part-2-Screen-Capture-and-Facial-Recognition
