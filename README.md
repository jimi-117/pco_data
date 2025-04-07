# pco_data

# MongoDBの起動
sudo service mongodb start

# PostgreSQLの設定
sudo -u postgres psql
CREATE DATABASE food_db;
CREATE USER user WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE food_db TO user;