import os

from matplotlib import pyplot as plt
from sklearn.model_selection import LearningCurveDisplay

from python.algorithms.classification.naive_bayes import NaiveBayes
from python.algorithms.classification.naive_bayes_cuda_optimized import NaiveBayesCUDAOptimized
from python.models.data import DataLoaderFactory
from python.models.text_processor import TextProcessor




def learning_curve(model, x, y, feature_names):
    LearningCurveDisplay.from_estimator(
        estimator=model,
        X=x,
        y=y,
        fit_params={'feature_names': feature_names},
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        cv=5,
        scoring="accuracy",
        score_type="both",
        n_jobs=8,
        std_display_style='fill_between',
    )
    plt.title("Learning Curve - MultinomialNB")
    plt.grid(True)
    plt.show()


def main():
    loader = DataLoaderFactory.create_loader(
        'huggingface',
        dataset_name='papluca/language-identification',
    )
    model = NaiveBayesCUDAOptimized(alpha=0.001, use_gpu=True)
    processor = TextProcessor(ngram_range=(1, 3), max_features=5000)
    x_train, y_train = loader.load_data(loader.dataset_name, 'train')
    x_train_processed = processor.fit_transform(x_train)
    learning_curve(model, x_train_processed , y_train, processor.get_feature_names())


if __name__ == '__main__':
    main()
