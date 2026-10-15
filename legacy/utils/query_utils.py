import time
import sys
import os
import json
from typing import *
from tqdm import tqdm
import signal
from openai import OpenAI
from together import Together

def query_model(
        client: OpenAI|Together,
        model: str,
        input_prompt: str,
        system_prompt: str = None,
        chat: bool = True,
        logprobs: bool = False,
        echo: bool = False,
        stream: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> Dict:
    client_api = client.__class__.__name__
    # Set up the call configuration
    call_config = {
        'stream': stream,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'logprobs': int(logprobs) if client_api == 'Together' else logprobs
    }

    if chat:
        # Chat API requires a different configuration
        messages = [{"role": "user", "content": input_prompt}]
        messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages # Add the system prompt if provided
        call_config.update({'messages': messages})
    else:
        assert not system_prompt, "System prompt is only available in chat mode. Consider adding it to the input prompt."
        call_config.update({'prompt': input_prompt})

    if echo:
        assert logprobs, "Echo requires logprobs to be enabled"
        call_config.update({'echo': 'true'})
    
    # Call the model
    response = client.chat.completions.create(**call_config) if chat else client.completions.create(**call_config)

    # Extract the data
    response = response.choices[0]
    data = {'message': response.message.content if chat else response.text}

    # Extract reasoning if available
    if hasattr(response.message, 'reasoning_content'):
        data['reasoning_content'] = response.message.reasoning_content
    
    # Extract logprobs
    if logprobs:
        if client_api == 'Together':
            data.update({
                'tokens': response.logprobs.tokens,
                'logprobs': response.logprobs.token_logprobs
            })
        elif client_api == 'OpenAI':
            data.update({
                'tokens': [token.token for token in response.logprobs.content],
                'logprobs': [token.logprob for token in response.logprobs.content]
            })

    # Prepend the echo to the message
    if echo:
        data['tokens'] = response.prompt[0].logprobs.tokens + data['tokens']
        data['logprobs'] = response.prompt[0].logprobs.token_logprobs + data['logprobs']

    data = {k: v for k, v in data.items() if v is not None} # Remove None values
    return data

def run_experiment(
        input_prompts: List[str],
        models: List[str],
        results_path: str,
        intermediate_path: str = None,
        system_prompt: str = None,
        query_timeout: int = 60,  # Timeout in seconds for each model query
        rate_limit: float = 0.02,  # Time to sleep between queries
        **kwargs
    ) -> Dict:
    # Define a handler for the timeout
    def handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, handler)

    # Initialize Client APIs
    together_client = Together()
    openai_client = OpenAI()
    deepseek_client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))

    # Get available models ids
    together_models = [model.id for model in together_client.models.list()]
    openai_models = [model.id for model in openai_client.models.list()]
    deepseek_models = [model.id for model in deepseek_client.models.list()]
    all_models = together_models + openai_models + deepseek_models
    
    # Check if the models are available
    for model in models:
        if model not in all_models:
            raise ValueError(f"Model {model} not found in available models")

    # Initialize dictionary to store the data
    if not os.path.exists(results_path):
        json.dump({'data': {}}, open(results_path, 'w'))  # Create an empty file
    data = json.load(open(results_path, 'r'))
    
    # Remove models that have already been queried
    models = [model for model in models if model not in data['data'].keys()]        
    if not models:
        print('All models have already been queried')
        return data

    # Loop over models
    for idx, model in enumerate(models):
        print(f'_____ Model: {model} ({idx+1}/{len(models)}) _____')

        # Get the client for the model
        if model in together_models:
            client = together_client 
        elif model in openai_models:
            client = openai_client
        elif "deepseek" in model:
            client = deepseek_client
            kwargs['logprobs'] = False # Deepseek does not support logprobs
        
        # Load intermediate results (if available)
        model_responses = []
        if intermediate_path and os.path.exists(intermediate_path):
            model_responses = json.load(open(intermediate_path, 'r'))
            print(f"Resuming from {len(model_responses)} completed prompts")
        
        # Resume from last completed prompt
        prompts = input_prompts[len(model_responses):]
            
        # Input all the items to the model
        for input_prompt in tqdm(prompts, desc=f"Processing {model}"):
            try:
                signal.alarm(query_timeout)  # Start the timeout clock
                response = query_model(client, model, input_prompt, system_prompt, **kwargs)
                model_responses.append(response)
                signal.alarm(0)  # Reset the alarm once the query completes
                time.sleep(rate_limit) # Sleep for a bit to avoid rate limiting
            except TimeoutError:
                print(f"Query for {model} timed out!")
                break
            except json.JSONDecodeError as e:
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    print("Critical Error: Empty or invalid JSON response. Stopping script.")
                    sys.exit(1)
            except Exception as e:
                print(f"Calling {model} failed: {str(e)}")
                break
                
            if intermediate_path:
                json.dump(model_responses, open(intermediate_path, 'w'))
        
        # Save results
        if len(model_responses) == len(input_prompts):
            data['data'][model] = model_responses
            json.dump(data, open(results_path, 'w'), indent=4)
            
            # Remove intermediate file
            if intermediate_path:
                os.remove(intermediate_path)

    return data