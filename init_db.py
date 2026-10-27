
from backend.app.db.database import Base, engine
from backend.app.db import models  # Registers SQLAlchemy models


def main():
    Base.metadata.create_all(bind=engine)
    print("FedLearn database tables created successfully!")


if __name__ == "__main__":
    main()
