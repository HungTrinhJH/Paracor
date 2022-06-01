# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Endata(models.Model):
    lang = models.CharField(max_length=2)
    sentence_id = models.CharField(max_length=6)
    word_id = models.CharField(max_length=2)
    word = models.CharField(max_length=100)
    lemma = models.CharField(max_length=100)
    links = models.CharField(max_length=100, blank=True, null=True)
    morpho = models.CharField(max_length=100, blank=True, null=True)
    pos = models.CharField(max_length=10, blank=True, null=True)
    phrase = models.CharField(max_length=100, blank=True, null=True)
    grm = models.CharField(max_length=100, blank=True, null=True)
    ner = models.CharField(max_length=20, blank=True, null=True)
    semantic = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'EnData'


class Ensentence(models.Model):
    sentence_id = models.CharField(max_length=6)
    sentence = models.CharField(max_length=1000)

    class Meta:
        managed = False
        db_table = 'EnSentence'


class Enstatistics(models.Model):
    word = models.CharField(primary_key=True, max_length=100)
    count = models.IntegerField(blank=True, null=True)
    pos = models.CharField(max_length=10)
    ner = models.CharField(max_length=20)
    semantic = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'EnStatistics'
        unique_together = (('word', 'pos', 'ner', 'semantic'),)


class Totalstatistics(models.Model):
    lang = models.CharField(primary_key=True, max_length=2)
    totaltoken = models.IntegerField(db_column='totalToken', blank=True, null=True)  # Field name made lowercase.
    totalsentence = models.IntegerField(db_column='totalSentence', blank=True, null=True)  # Field name made lowercase.
    totalword = models.IntegerField(db_column='totalWord', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'TotalStatistics'


class User(models.Model):
    username = models.CharField(max_length=100, blank=True, null=True)
    password = models.CharField(max_length=100, blank=True, null=True)
    role = models.CharField(max_length=5, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'User'


class Vndata(models.Model):
    lang = models.CharField(max_length=2)
    sentence_id = models.CharField(max_length=6)
    word_id = models.CharField(max_length=2)
    word = models.CharField(max_length=100)
    lemma = models.CharField(max_length=100)
    links = models.CharField(max_length=100, blank=True, null=True)
    morpho = models.CharField(max_length=100, blank=True, null=True)
    pos = models.CharField(max_length=10, blank=True, null=True)
    phrase = models.CharField(max_length=100, blank=True, null=True)
    grm = models.CharField(max_length=100, blank=True, null=True)
    ner = models.CharField(max_length=20, blank=True, null=True)
    semantic = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'VnData'


class Vnsentence(models.Model):
    sentence_id = models.CharField(max_length=6)
    sentence = models.CharField(max_length=1000)

    class Meta:
        managed = False
        db_table = 'VnSentence'


class Vnstatistics(models.Model):
    word = models.CharField(primary_key=True, max_length=100)
    count = models.IntegerField(blank=True, null=True)
    pos = models.CharField(max_length=10)
    ner = models.CharField(max_length=20)
    semantic = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'VnStatistics'
        unique_together = (('word', 'pos', 'ner', 'semantic'),)


class DjangoMigrations(models.Model):
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'
