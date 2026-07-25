# FMNIST Classifier

Классификатор изображений одежды из датасета Fashion-MNIST с использованием PyTorch и FastAPI.

## Цель проекта
Демонстрация навыков построени production-ready ML-сервиса: обучение модели, упаковка в Docker, API.

## Модель
- Архитектура: сверточная нейросеть на PyTorch
- Метрика: accuracy на тестовой выборке — 88.92%
- Датасет: Fashion-MNIST (70,000 изображений 28x28)

## Запуск
\`\`\`bash
docker-compose up
\`\`\`
После запуска API доступно по адресу http://localhost:8000/docs

## Структура проекта
- \`src/ml\` — обучение и инференс модели
- \`src/api\` — FastAPI-эндпоинты
- \`Dockerfile\`, \`docker-compose.yml\` — контейнеризация