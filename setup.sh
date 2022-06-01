sudo mkdir /var/www/html/admin
sudo mkdir /var/www/html/user
sudo mkdir /var/www/html/server
cd frontend/concordance-user/
npm install
npm run build
sudo cp -rf build /var/www/html/user
cd ../concordance-admin/admin-webapp/
npm install
npm run build
sudo cp -rf build /var/www/html/admin
cd ../../
sudo cp -rf backend/concordance /var/www/html/server
sudo apt install libapache2-mod-wsgi-py3 
sudo a2enmod wsgi
sudo install mysql-server
sudo mysql_secure_installation
sudo mysql -e "
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'admin123';
FLUSH PRIVILEGES;
exit;

"
mysql -uroot -p admin123 -e "
CREATE DATABASE concordance;

use concordance;

CREATE TABLE IF NOT EXISTS EnData (
    id int auto_increment PRIMARY KEY,
    lang varchar(2) NOT NULL,
    sentence_id varchar(6) NOT NULL,
    word_id varchar(2) NOT NULL,
    word nvarchar(100) NOT NULL,
    lemma nvarchar(100) NOT NULL,
    links varchar(100),
    morpho varchar(100),
    pos varchar(10),
    phrase varchar(100),
    grm varchar(100),
    ner varchar(20),
    semantic varchar(100),
    INDEX (word, lemma, pos, ner, sentence_id)
);

CREATE TABLE IF NOT EXISTS VnData (
    id int auto_increment PRIMARY KEY,
    lang varchar(2) NOT NULL,
    sentence_id varchar(6) NOT NULL,
    word_id varchar(2) NOT NULL,
    word nvarchar(100) NOT NULL,
    lemma nvarchar(100) NOT NULL,
    links varchar(100),
    morpho varchar(100),
    pos varchar(10),
    phrase varchar(100),
    grm varchar(100),
    ner varchar(20),
    semantic varchar(100),
    INDEX (word, lemma, pos, ner, sentence_id)
);

CREATE TABLE IF NOT EXISTS EnSentence (
    id int auto_increment PRIMARY KEY,
    sentence_id varchar(6) NOT NULL,
    sentence nvarchar(1000) NOT NULL,
    INDEX (sentence_id)
);

CREATE TABLE IF NOT EXISTS VnSentence (
    id int auto_increment PRIMARY KEY,
    sentence_id varchar(6) NOT NULL,
    sentence nvarchar(1000) NOT NULL,
    INDEX (sentence_id)
);

CREATE TABLE IF NOT EXISTS EnStatistics (
    word nvarchar(100) NOT NULL,
    count INTEGER,
    pos varchar(10),
    ner varchar(20),
    semantic varchar(100),
    PRIMARY KEY (word, pos, ner, semantic)
);

CREATE TABLE IF NOT EXISTS VnStatistics (
    word nvarchar(100) NOT NULL,
    count INTEGER,
    pos varchar(10),
    ner varchar(20),
    semantic varchar(100),
    PRIMARY KEY (word, pos, ner, semantic)
);

CREATE TABLE IF NOT EXISTS TotalStatistics (
    lang varchar(2) NOT NULL,
    totalToken INTEGER,
    totalSentence INTEGER,
    totalWord INTEGER,
    PRIMARY KEY (lang)
);

INSERT TotalStatistics VALUES ("en", 0,0,0), ("vn", 0,0,0);

CREATE TRIGGER after_EnData_insert
        AFTER INSERT ON EnData
        FOR EACH ROW
    INSERT INTO EnStatistics
    SET word = LOWER(NEW.word),
        count = 1,
        pos = NEW.pos,
        ner = NEW.ner,
        semantic = NEW.semantic
    ON DUPLICATE KEY UPDATE count = count + 1;


CREATE TRIGGER after_EnData_insert1
        AFTER INSERT ON EnData
        FOR EACH ROW
    UPDATE TotalStatistics SET totalToken = totalToken + 1 WHERE lang = "en";


CREATE TRIGGER after_VnData_insert
        AFTER INSERT ON VnData
        FOR EACH ROW
    INSERT INTO VnStatistics
    SET word = LOWER(NEW.word),
        count = 1,
        pos = NEW.pos,
        ner = NEW.ner,
        semantic = NEW.semantic
    ON DUPLICATE KEY UPDATE count = count + 1;


CREATE TRIGGER after_VnData_insert1
        AFTER INSERT ON VnData
        FOR EACH ROW
    UPDATE TotalStatistics SET totalToken = totalToken + 1 WHERE lang = "vn";

CREATE TRIGGER after_VnSentence_insert
        AFTER INSERT ON VnSentence
        FOR EACH ROW
    UPDATE TotalStatistics SET totalSentence = totalSentence + 1 WHERE lang = "vn";

CREATE TRIGGER after_EnSentence_insert
        AFTER INSERT ON EnSentence
        FOR EACH ROW
    UPDATE TotalStatistics SET totalSentence = totalSentence + 1 WHERE lang = "en";

CREATE TRIGGER after_EnStatistics_insert
        AFTER INSERT ON EnStatistics
        FOR EACH ROW
    UPDATE TotalStatistics SET totalWord = totalWord + 1 WHERE lang = "en";

CREATE TRIGGER after_VnStatistics_insert
        AFTER INSERT ON VnStatistics
        FOR EACH ROW
    UPDATE TotalStatistics SET totalWord = totalWord + 1 WHERE lang = "vn";

delimiter //
CREATE TRIGGER after_EnData_update
        AFTER UPDATE ON EnData
        FOR EACH ROW
    BEGIN
    DECLARE temp varchar(1000);
    DECLARE num INT;
    SET temp = (SELECT GROUP_CONCAT(word SEPARATOR " ") FROM EnData where sentence_id = OLD.sentence_id order by word_id);
    UPDATE EnSentence SET sentence = temp where sentence_id = OLD.sentence_id;
    INSERT INTO EnStatistics
    SET word = NEW.word,
        count = 1,
        pos = NEW.pos,
        ner = NEW.ner,
        semantic = NEW.semantic
    ON DUPLICATE KEY UPDATE count = count + 1;
    set num = (SELECT count from EnStatistics where word = OLD.word and pos = OLD.pos and ner= OLD.ner and semantic= OLD.semantic);
    if num=1 THEN 
        DELETE FROM EnStatistics where word = OLD.word and pos = OLD.pos and ner= OLD.ner and semantic= OLD.semantic;
    ELSE 
        UPDATE EnStatistics SET count = count - 1 where word = OLD.word and pos = OLD.pos and ner= OLD.ner and semantic= OLD.semantic;
    END IF;
    END;//

CREATE TRIGGER after_delete_EnStatistics 
        AFTER DELETE ON EnStatistics
        FOR EACH ROW
    UPDATE TotalStatistics SET totalWord = totalWord - 1 where lang='en';//



CREATE TRIGGER after_VnData_update
        AFTER UPDATE ON VnData
        FOR EACH ROW
    BEGIN
    DECLARE temp varchar(1000);
    DECLARE num INT;
    SET temp = (SELECT GROUP_CONCAT(word SEPARATOR " ") FROM VnData where sentence_id = OLD.sentence_id order by word_id);
    UPDATE VnSentence SET sentence = temp where sentence_id = OLD.sentence_id;
    INSERT INTO VnStatistics
    SET word = NEW.word,
        count = 1,
        pos = NEW.pos,
        ner = NEW.ner,
        semantic = NEW.semantic
    ON DUPLICATE KEY UPDATE count = count + 1;
    set num = (SELECT count from VnStatistics where word = OLD.word and pos = OLD.pos and ner= OLD.ner and semantic= OLD.semantic);
    if num=1 THEN 
        DELETE FROM VnStatistics where word = OLD.word and pos = OLD.pos and ner= OLD.ner and semantic= OLD.semantic;
    ELSE 
        UPDATE VnStatistics SET count = count - 1 where word = OLD.word and pos = OLD.pos and ner= OLD.ner and semantic= OLD.semantic;
    END IF;
    END;//

CREATE TRIGGER after_delete_VnStatistics 
        AFTER DELETE ON VnStatistics
        FOR EACH ROW
    UPDATE TotalStatistics SET totalWord = totalWord - 1 where lang='vn';//

CREATE TABLE IF NOT EXISTS User (
    id int auto_increment PRIMARY KEY,
    username varchar(100),
    password varchar(100),
    role ENUM('admin','user') default 'user'
);//

INSERT INTO User(username,password,role) VALUES('admin','md5$ITNNUQOrkzOI$cc83623eb33cf54c161c4faf3515420b','admin');//

exit;//

"
#pass admin is 123456
pip3 install django djangorestframework django-cors-headers pyjwt mysqlclient
