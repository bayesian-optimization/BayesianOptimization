import numpy as np
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import PickleType

Base = declarative_base()


class TargetSpaceTable(Base):
    __tablename__ = 'target_space_table'
    id = Column(Integer, primary_key=True)
    keys = Column(PickleType, nullable=False)
    X = Column(PickleType, nullable=False)
    Y = Column(PickleType, nullable=False)


class DBManager:
    def __init__(self, conn_str):
        self.engine = create_engine(conn_str, echo=False)
        self.table_name = 'target_space_table'

        # if table doesn't exist, create.
        if not self.engine.dialect.has_table(self.engine, self.table_name):
            Base.metadata.create_all(self.engine)

        Base.metadata.bind = self.engine
        self.DBSession = sessionmaker(bind=self.engine)
        self.session = self.DBSession()

    def save(self, keys, X, Y):
        # find the row to be saved
        target_row = self.session.query(TargetSpaceTable).first()

        # if no historical data saved
        if target_row is None:
            new_record = TargetSpaceTable(keys=keys, X=X, Y=Y)
            self.session.add(new_record)
            self.session.commit()
        else:
            target_row.keys = keys
            target_row.X = X
            target_row.Y = Y
            self.session.commit()

    def load(self):
        # find the row to be loaded
        target_row = self.session.query(TargetSpaceTable).first()

        # if no historical data saved
        if target_row is None:
            return None
        else:
            init_points = dict(zip(target_row.keys, np.asarray(target_row.X).T.tolist()))
            init_points['target'] = target_row.Y.tolist()
            return init_points

    def clear(self):
        self.session.query(TargetSpaceTable).delete()



