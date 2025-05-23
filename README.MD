# Film Genre Prediction

This project predicts film genres using machine learning.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/film_genre_prediction.git
cd film_genre_prediction
```

### 2. Create a Virtual Environment

#### On Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

You are now ready to start working on the project!

## How to Run

To process the data and create training datasets for the film genre prediction model, follow these steps in order:

### 1. Process Street Data

Run the `process_street.ipynb` notebook first to prepare geographical data:

```bash
jupyter notebook src/process_street.ipynb
```

This notebook processes street and district information from Đà Nẵng city that will be used for feature engineering.

### 2. Clean Raw Data

Next, run the data cleaning notebook to prepare the raw ticket and film data:

```bash
jupyter notebook src/raw_cleaning.ipynb
```

This notebook performs several important steps:
- Merges ticket data with film metadata
- Cleans and normalizes film titles
- Maps film titles to their corresponding genres
- Exports the cleaned data to `processed_data/cleaned_raw_data.csv`

### 3. Generate Training Data

Finally, run the training data generation notebook:

```bash
jupyter notebook src/generate_training_data.ipynb
```

This notebook:
- Processes the cleaned data into features suitable for machine learning
- Creates both ticket-level and customer-level data representations
- Transforms categorical features into probabilities
- Converts film genres into probability distributions for each customer
- Performs train/test splits and saves the prepared datasets to the `training_data/` directory

The final datasets in the `training_data/` folder are ready to be used for model training.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## References
- [1] [Danh sách tên các con đường tại thành phố Đà Nẵng" - Congdulieu.vn](https://congdulieu.vn/dulieuchitiet/1037328#)
- [2] Đơn vị hành chính. (n.d.). [https://danhmuchanhchinh.gso.gov.vn/](https://danhmuchanhchinh.gso.gov.vn/)
- [3] Vu, T. (2017, May 24). Bài 24: Neighborhood-Based Collaborative Filtering. Tiep Vu’s Blog. https://machinelearningcoban.com/2017/05/24/collaborativefiltering/