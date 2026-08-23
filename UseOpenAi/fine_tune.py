from openai import OpenAI  # Library to interact with OpenAI API

# Initialize OpenAI client to communicate with OpenAI services
client = OpenAI()

# Upload the test.jsonl file to OpenAI for fine-tuning purposes
file_response = client.files.create(file=open("test.jsonl","rb"),
                                    purpose="fine-tune")
print(f"Uploaded file: {file_response}")

# List all files currently uploaded to OpenAI to verify the upload was successful
files = client.files.list()
print("All uploaded files:")
for file in files.data:
    print(f"  {file.id}: {file.filename} ({file.purpose})")

# Retrieve detailed information about the uploaded file for confirmation
file_info = client.files.retrieve(file_response.id)
print(f"\nRetrieved file info: {file_info}")

# Create a fine-tuning job with specific parameters
fine_tune_job = client.fine_tuning.jobs.create(
    model="gpt-3.5-turbo",  # Base model to fine-tune
    training_file=file_info.id,  # File containing training data
    hyperparameters={  # Hyperparameters for the fine-tuning process
        "n_epochs": 1  # Number of training iterations
    },
    validation_file=file_response.id  # File containing validation data
)

print(f"\nFine-tuning job created: {fine_tune_job}")

# List recent fine-tuning jobs to show current state of all jobs
jobs = client.fine_tuning.jobs.list(limit=10)
print("\nRecent fine-tuning jobs:")
for job in jobs.data:
    print(f"  {job.id}: {job.status} - {job.model}")

# Check the status of our specific fine-tuning job to monitor progress
job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
print(f"\nJob status: {job_status}")

# Get recent events from the fine-tuning process to track progress
events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id=fine_tune_job.id,
    limit=5
)
print("\nRecent job events:")
for event in events.data:
    print(f"  {event.created_at}: {event.message}")

# Test the fine-tuned model with a sample input to verify it works correctly
completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:bis::9tt19es9",  # The fine-tuned model identifier
    messages=[
        {"role": "user", "content": "Classify the articles into these categories: business, entertainment, politics, sport, tech. A new mobile phone is launched"}
    ]
)
print(f"\nTest classification result: {completion.choices[0].message.content}")