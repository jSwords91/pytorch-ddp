import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataConfig:
    movies_file_path: str = "movielens_small/movies.csv"
    ratings_file_path: str = "movielens_small/ratings.csv"
    movie_field: str = "movieId"
    user_field: str = "userId"
    rating_field: str = "rating"
    matrix_fields: list = field(default_factory=lambda: ["userId", "movieId", "rating"])


def load_data(config: DataConfig) -> pd.DataFrame:
    movies_df = pd.read_csv(config.movies_file_path)
    ratings_df = pd.read_csv(config.ratings_file_path)
    return movies_df, ratings_df


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
        train_loader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True)
        return train_loader, len(dataset_train), val_loader, len(dataset_val)
    else:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        ), len(dataset)


class NeuralRecSys(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_user_factors: int,
        num_item_factors: int,
    ):
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
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_every: int,
    ):
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
            users, movies, ratings = (
                users.to(self.device),
                movies.to(self.device),
                ratings.to(self.device),
            )
            loss = self._run_batch(users, movies, ratings)
            total_loss += loss
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for users, movies, ratings in self.val_loader:
                users, movies, ratings = (
                    users.to(self.device),
                    movies.to(self.device),
                    ratings.to(self.device),
                )
                predictions = self.model(users, movies).squeeze()
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def _show_progress(self, epoch, train_loss, val_loss):
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}"
        )

    def _save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
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
            self._show_progress(epoch, train_loss, val_loss)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_loss, val_loss)
        return train_losses, val_losses


def load_train_objs(num_users, num_movies, user_factors, item_factors):
    model = NeuralRecSys(
        num_users=num_users,
        num_items=num_movies,
        num_user_factors=user_factors,
        num_item_factors=item_factors,
    )

    model.apply(init_weights)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.2)
    return model, criterion, optimizer


def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    return None


def set_seed(seed: int = 12345):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main(device: str, total_epochs: int, save_every: int, batch_size: int):

    set_seed()

    dataConfig = DataConfig()
    movies_df, ratings_df = load_data(dataConfig)
    df = ratings_df.merge(movies_df)[dataConfig.matrix_fields]

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df[dataConfig.user_field] = user_encoder.fit_transform(df[dataConfig.user_field])
    df[dataConfig.movie_field] = movie_encoder.fit_transform(df[dataConfig.movie_field])

    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)

    dataset = MovieRatingsDataset(
        torch.tensor(df[dataConfig.user_field].values),
        torch.tensor(df[dataConfig.movie_field].values),
        torch.tensor(df[dataConfig.rating_field].values, dtype=torch.float),
    )
    train_loader, train_len, val_loader, val_len = prepare_data_loaders(
        dataset, split=0.10, batch_size=batch_size
    )

    user_factors, item_factors = embedding_size_heuristic(num_users), embedding_size_heuristic(num_movies)

    model, criterion, optimizer = load_train_objs(
        num_users, num_movies, user_factors, item_factors
    )

    print(model)
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, criterion, device, save_every
    )
    train_losses, val_losses = trainer.train(total_epochs)
    plot_losses(train_losses, val_losses, save_path="loss_plot.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural Recommendation System Training"
    )
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Input batch size on each device (default: 256)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    main(device, args.total_epochs, args.save_every, args.batch_size)
