import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    movies_df = pd.read_csv("movielens_small/movies.csv")
    ratings_df = pd.read_csv("movielens_small/ratings.csv")
    return movies_df, ratings_df

def calculate_popularity(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, damping_factor: int = 5) -> pd.DataFrame:
    rating_stats = ratings_df.groupby("movieId")["rating"].agg(['count', 'mean'])
    global_mean = ratings_df["rating"].mean()

    damped_mean = (rating_stats['count'] * rating_stats['mean'] + damping_factor * global_mean) / (rating_stats['count'] + damping_factor)

    movies_df = movies_df.join(rating_stats.rename(columns={'count': 'num_ratings', 'mean': 'mean_rating'}), on='movieId')
    movies_df['damped_mean_rating'] = movies_df['movieId'].map(damped_mean)

    return movies_df

def embedding_size_heuristic(n_cat: int) -> int:
    return min(600, round(1.6 * n_cat**0.56))

class MovieRatingsDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

def prepare_data_loaders(dataset: Dataset, split: float = None, batch_size: int = 128):
    if split:
        train_size = int(len(dataset) * (1 - split))
        val_size = len(dataset) - train_size
        dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True)
        return train_loader, len(dataset_train), val_loader, len(dataset_val)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True), len(dataset)

class NeuralRecSys(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_user_factors: int, num_item_factors: int):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, num_user_factors)
        self.item_factors = nn.Embedding(num_items, num_item_factors)
        self.lin = nn.Linear(num_user_factors + num_item_factors, 1)

    def forward(self, user, item):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        return self.lin(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, save_every: int):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_every = save_every

    def _run_batch(self, users, movies, ratings):
        self.optimizer.zero_grad()
        predictions = self.model(users, movies).squeeze()
        loss = self.criterion(predictions, ratings)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for users, movies, ratings in self.train_loader:
            users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
            loss = self._run_batch(users, movies, ratings)
            total_loss += loss
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for users, movies, ratings in self.val_loader:
                users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
                predictions = self.model(users, movies).squeeze()
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, f"recsys_checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved at epoch {epoch}")

    def train(self, num_epochs: int):
        train_losses, val_losses = [], []
        for epoch in range(num_epochs):
            train_loss = self._run_epoch(epoch)
            val_loss = self._validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)
        return train_losses, val_losses

def main(device, total_epochs, save_every, batch_size):
    # Set random seed for reproducibility
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and preprocess data
    movies_df, ratings_df = load_data()
    movies_df = calculate_popularity(movies_df, ratings_df, damping_factor=8)
    
    df = ratings_df.merge(movies_df)[['userId', 'movieId', 'rating']].sample(n=50_000)
    
    # Encode user and movie IDs
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['movieId'] = movie_encoder.fit_transform(df['movieId'])

    # Create dataset and data loaders
    dataset = MovieRatingsDataset(
        torch.tensor(df['userId'].values),
        torch.tensor(df['movieId'].values),
        torch.tensor(df['rating'].values, dtype=torch.float)
    )
    train_loader, train_len, val_loader, val_len = prepare_data_loaders(dataset, split=0.10, batch_size=batch_size)

    # Initialize model
    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    user_factors = embedding_size_heuristic(num_users)
    item_factors = embedding_size_heuristic(num_movies)

    model = NeuralRecSys(num_users=num_users, num_items=num_movies, num_user_factors=user_factors, num_item_factors=item_factors)
    model.apply(init_weights)
    print(model)

    # Initialize optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.2)

    # Initialize trainer and start training
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, save_every)
    train_losses, val_losses = trainer.train(total_epochs)

    # Here you could add code to plot the losses or perform final model evaluation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Recommendation System Training')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device (default: 128)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main(device, args.total_epochs, args.save_every, args.batch_size)