import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, util
import json
# --- KNOWLEDGE BASE CLASS ---
# This class handles loading and searching the knowledge base.
class KnowledgeBase:
    """
    Handles loading the FAISS vector index and the JSON mapping file.
    """
    def __init__(self, index_path, mapping_path, model_name='all-MiniLM-L6-v2'):
        print(f"Loading knowledge base from: {index_path}")
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Database files not found. Please ensure '{index_path}' and "
                f"'{mapping_path}' exist. You may need to run the "
                f"'build_database.py' script first."
            )
        
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
        print("✅ Knowledge base loaded successfully.")

# --- AGENT BRAINS (NEURAL NETWORKS) ---
# These small networks are what the agents use to learn the new language.
class EncoderNet(nn.Module):
    """A simple network for the Communication Agent to transform embeddings."""
    def __init__(self, input_dim, output_dim):
        super(EncoderNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.tanh(self.fc1(x))

class DecoderNet(nn.Module):
    """A simple network for the Retrieval Agent to interpret messages."""
    def __init__(self, input_dim, output_dim):
        super(DecoderNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)

# --- AGENT LOGIC WRAPPERS ---
class CommunicationEncoder:
    """The core logic for the Communication Agent during training."""
    def __init__(self, device, model_name='all-MiniLM-L6-v2'):
        self.device = device
        self.base_model = SentenceTransformer(model_name, device=self.device)
        embedding_dim = self.base_model.get_sentence_embedding_dimension()
        self.network = EncoderNet(input_dim=embedding_dim, output_dim=128).to(self.device)
        print("✅ Communication Encoder (CA) initialized with a trainable network.")

    def create_message(self, user_query: str):
        """Encodes a query into an initial embedding and a learned emergent message."""
        with torch.no_grad():
            initial_embedding = self.base_model.encode([user_query], convert_to_tensor=True, device=self.device)
        emergent_message = self.network(initial_embedding)
        return initial_embedding, emergent_message

class RetrievalDecoder:
    """The core logic for the Retrieval Agent during training."""
    def __init__(self, knowledge_base: KnowledgeBase, device):
        self.kb = knowledge_base
        self.device = device
        embedding_dim = self.kb.model.get_sentence_embedding_dimension()
        self.network = DecoderNet(input_dim=128, output_dim=embedding_dim).to(self.device)
        print("✅ Retrieval Decoder (RA) initialized with a trainable network.")

    def process_message(self, message_vector: torch.Tensor):
        """Decodes an emergent message into a search vector to find a document."""
        search_vector = self.network(message_vector)
        # FAISS (CPU version) runs on the CPU, so we move the vector back before searching
        numpy_vector = search_vector.detach().cpu().numpy().astype('float32')
        distances, ids = self.kb.index.search(numpy_vector, 1)
        
        best_id = ids[0][0]
        if best_id == -1: return None, None
        
        retrieved_chunk = self.kb.mapping.get(str(best_id))
        return search_vector, retrieved_chunk

# --- DATA LOADER ---
def load_training_data(file_path: str, num_samples: int = 100) -> list:
    """Loads a sample of queries from a ticket CSV file."""
    print(f"\nLoading training data from {file_path}...")
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
        query_column = 'body' if 'body' in df.columns else 'Complaint'
        if query_column not in df.columns:
            print(f"Warning: Could not find a query column in {file_path}. Skipping.")
            return []
        
        df.dropna(subset=[query_column], inplace=True)
        queries = df[query_column].tolist()
        # Return a random sample to keep training fast for this example
        return np.random.choice(queries, size=min(num_samples, len(queries)), replace=False).tolist()
    except Exception as e:
        print(f"Could not load data from {file_path}. Error: {e}")
        return []

# --- THE MAIN TRAINING ORCHESTRATOR ---
if __name__ == '__main__':
    # --- Setup ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    DATABASE_DIR = "dataset"
    INDEX_PATH = os.path.join(DATABASE_DIR, "nexa_corp.index")
    MAPPING_PATH = os.path.join(DATABASE_DIR, "nexa_corp_mapping.json")
    
    NUM_EPOCHS = 100

    try:
        # 1. Initialize all components
        kb = KnowledgeBase(index_path=INDEX_PATH, mapping_path=MAPPING_PATH)
        comm_agent = CommunicationEncoder(device=DEVICE)
        retrieval_agent = RetrievalDecoder(knowledge_base=kb, device=DEVICE)

        # 2. Setup the optimizer and loss function
        all_params = list(comm_agent.network.parameters()) + list(retrieval_agent.network.parameters())
        optimizer = optim.Adam(all_params, lr=0.001)
        # Cosine similarity loss: pushes the two vectors to be as similar as possible.
        loss_fn = nn.CosineEmbeddingLoss()
        
        # 3. Load training data
        # Load queries from multiple files
        training_queries = []
        ticket_files = [
            "dataset-tickets-multi-lang-4-20k.csv",
            "dataset-tickets-multi-lang-4-20k-2.csv",
            "dataset-tickets-multi-lang-4-20k-3.csv",
            "nexacorp_tickets.xlsx"
        ]
        for file_name in ticket_files:
            queries = load_training_data(
            os.path.join(DATABASE_DIR, file_name),
            num_samples=333  # Adjust sample size per file if needed
            )
            training_queries.extend(queries)
        # Optionally, shuffle and limit total queries
        np.random.shuffle(training_queries)
        training_queries = training_queries[:1000]
        if not training_queries:
            raise ValueError("No training data could be loaded. Aborting.")

        print("\n===================================================")
        print(f"      🚀 Starting Emergent Language Training ({NUM_EPOCHS} Epochs) on {DEVICE}      ")
        print("===================================================")

        # 4. Run the training loop
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
            
            for i, query in enumerate(training_queries):
                optimizer.zero_grad()

                # Agents perform their actions
                initial_embedding, message_vector = comm_agent.create_message(user_query=query)
                search_vector, result_chunk = retrieval_agent.process_message(message_vector=message_vector)
                
                # Skip this training step if no document was found
                if search_vector is None:
                    continue

                # The target is a tensor of 1s, telling the loss function we want the vectors to be similar.
                target = torch.ones(initial_embedding.size(0)).to(DEVICE)
                
                # Calculate the loss and update the networks
                loss = loss_fn(initial_embedding, search_vector, target)
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()

                if (i + 1) % 50 == 0:
                    print(f"  ...processed {i+1}/{len(training_queries)} queries...")

            print(f"Epoch {epoch + 1} complete. Average Loss: {total_loss / len(training_queries):.4f}")

        # 5. Save the trained models
        print("\n💾 Saving trained agent networks...")
        torch.save(comm_agent.network.state_dict(), "encoder_net.pth")
        torch.save(retrieval_agent.network.state_dict(), "decoder_net.pth")
        print("✅ Models saved as 'encoder_net.pth' and 'decoder_net.pth'.")
        
        print(f"\n🏁 Training complete.")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}\nPlease run 'build_database.py' first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
