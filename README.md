## Project outline: 
Scalable and flexible chatbot architecture, with separate microservices for intent and entity recognition and text generation
	
### Kubeflow pipeline components
- Use Langchain to preprocess and label custom data (complaints from users of a financial product/service) for intent and entity recognition
- Seldon to deploy Langchain agent as a microservice on Kubernetes
- TorchServe to deploy GPT models as microservices alongside the Langchain agent
- Connect frontend (Streamlit) to the kubernetes service to serve the application


### GCR - Containerize everything for modularity 
- LangChain container for pre-processing & labeling
- Seldon container for deploying Langchain agent as a microservice on Kubernetes
- TorchServe container for deploying GPT models (responses) as microservices alongside Langchain agent
- Streamlit container for frontend on Kubernetes 

If time permits: MLflow for versioning, & CI/CD pipeline for automation
Integrate LangChain with Seldon/TFServing/TorchServe
