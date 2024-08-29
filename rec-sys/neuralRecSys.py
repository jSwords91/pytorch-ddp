import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass

@dataclass
class DataConfig:
    movie_file_path: str = "movielens_small/movies.csv"
    rating_file_path: str = "movielens_small/ratings.csv"
    movie_column_name: str = "movieId"
    user_column_name: str = "userId"
    rating_column_name: str = "rating"

@dataclass
class TrainingConfig:
    total_epochs: int = 10
    save_every: int = 5
    batch_size: int = 256


class MovieRatingsDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.users[idx], self.movies[idx]), self.ratings[idx]

class NeuralRecSys(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_user_factors: int, num_item_factors: int):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, num_user_factors)
        self.item_factors = nn.Embedding(num_items, num_item_factors)
        self.lin = nn.Linear(num_user_factors + num_item_factors, 1)

    def forward(self, input):
        user, item = input
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        return self.lin(x)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.criterion = nn.MSELoss()

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output.squeeze(), targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0][0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.model.train()
        train_loss = 0.0
        for source, targets in self.train_data:
            source = (source[0].to(self.gpu_id), source[1].to(self.gpu_id))
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            train_loss += loss
        return train_loss / len(self.train_data)

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for source, targets in self.val_data:
                source = (source[0].to(self.gpu_id), source[1].to(self.gpu_id))
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                loss = self.criterion(output.squeeze(), targets)
                val_loss += loss.item()
        return val_loss / len(self.val_data)

    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }
        PATH = f"./checkpoints/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            train_loss = self._run_epoch(epoch)
            val_loss = self._validate()
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def embedding_size_heuristic(n_cat: int) -> int:
    """Rule of thumb to pick embedding size corresponding to `n_cat` (number of categories). Used in the fastai library"""
    return min(600, round(1.6 * n_cat**0.56))

def load_train_objs(data_config: DataConfig):
    # Load and preprocess data
    movies_df = pd.read_csv(data_config.movie_file_path)
    ratings_df = pd.read_csv(data_config.rating_file_path)
    df = ratings_df.merge(movies_df)[[data_config.user_column_name, data_config.movie_column_name, data_config.rating_column_name]]

    # Encode user and movie IDs so they are contiguous integers
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df[data_config.user_column_name] = user_encoder.fit_transform(df[data_config.user_column_name])
    df[data_config.movie_column_name] = movie_encoder.fit_transform(df[data_config.movie_column_name])

    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)

    dataset = MovieRatingsDataset(
        torch.tensor(df[data_config.user_column_name].values, dtype=torch.long),
        torch.tensor(df[data_config.movie_column_name].values, dtype=torch.long),
        torch.tensor(df[data_config.rating_column_name].values, dtype=torch.float)
    )

    user_factor_size, movie_factor_size = embedding_size_heuristic(num_users), embedding_size_heuristic(num_movies)

    model = NeuralRecSys(num_users=num_users, num_items=num_movies, num_user_factors=user_factor_size, num_item_factors=movie_factor_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, split: float = 0.2):

    train_size = int((1 - split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )
    return train_loader, val_loader

def show_device_details() -> None:
    print(f"GPU DETAILS:")
    print("Is CUDA available: ", torch.cuda.is_available())
    print("Number of CUDA devices: ", torch.cuda.device_count())
    print("Current CUDA device: ", torch.cuda.current_device())
    print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))


def main(device: int, training_config: TrainingConfig, data_config: DataConfig):
    dataset, model, optimizer = load_train_objs(data_config)
    train_data, val_data = prepare_dataloader(dataset, training_config.batch_size)
    trainer = Trainer(model, train_data, val_data, optimizer, device, training_config.save_every)
    trainer.train(training_config.total_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Recommendation System Training')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', type=int, help='Input batch size on each device')
    args = parser.parse_args()

    if torch.cuda.is_available():
        show_device_details()

    device = 0  # shorthand for cuda:0
    data_config = DataConfig()
    training_config = TrainingConfig()

    # Override default config with command line arguments if provided
    if args.total_epochs is not None:
        training_config.total_epochs = args.total_epochs
    if args.save_every is not None:
        training_config.save_every = args.save_every
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size

    main(device, training_config, data_config)