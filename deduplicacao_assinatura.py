# Script PySpark para deduplicacao baseada em Assinatura de Registro com Jaccard e Levenshtein

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, when, length, levenshtein
from pyspark.sql.types import StringType, FloatType
import re

# Inicializa a sessão Spark
spark = SparkSession.builder.appName("DeduplicacaoAssinatura").getOrCreate()

# =========================== Funções Auxiliares ============================

def normalize_nome(nome):
    nome = nome.lower()
    nome = re.sub(r"\b(de|da|do|das|dos)\b", "", nome)
    nome = re.sub(r"\s+", " ", nome).strip()
    return nome

def extract_first_last(nome):
    partes = nome.strip().split()
    return partes[0], partes[-1] if len(partes) > 1 else (partes[0], partes[0])

def gerar_assinatura(nome_aluno, data_nasc, nome_mae, sexo):
    nome_aluno = normalize_nome(nome_aluno or "")
    nome_mae = normalize_nome(nome_mae or "")
    pn_a, un_a = extract_first_last(nome_aluno)
    pn_m, un_m = extract_first_last(nome_mae)
    return f"{pn_a} {un_a} {data_nasc} {pn_m} {un_m} {sexo}"

# UDFs
@udf(StringType())
def assinatura_udf(nome_aluno, data_nasc, nome_mae, sexo):
    return gerar_assinatura(nome_aluno, data_nasc, nome_mae, sexo)

@udf(FloatType())
def jaccard_udf(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return float(inter) / union if union != 0 else 0.0

# =========================== Processo Principal ============================

# Leitura do DataFrame df com registros
# df = df.withColumn("assinatura", assinatura_udf("nome", "data_nasc", "nome_mae", "sexo"))

# Comparação entre pares com mesmo CPF mas assinaturas diferentes
df_joined = df.alias("a").join(
    df.alias("b"),
    (col("a.cpf") == col("b.cpf")) &
    (col("a.id") < col("b.id")) & #<- Garante que o registro posterior seja comparado com um anterior
    (col("a.assinatura") != col("b.assinatura"))
)

# Cálculo de Jaccard, Levenshtein e score final
df_joined = df_joined \
    .withColumn("jaccard", jaccard_udf(col("a.assinatura"), col("b.assinatura"))) \
    .withColumn("lev_dist", levenshtein(col("a.assinatura"), col("b.assinatura")).cast("float")) \
    .withColumn("len_max", when(length(col("a.assinatura")) > length(col("b.assinatura")), length(col("a.assinatura"))).otherwise(length(col("b.assinatura")))) \
    .withColumn("lev_score", 1 - (col("lev_dist") / col("len_max"))) \
    .withColumn("score_final", 0.75 * col("jaccard") + 0.25 * col("lev_score"))

# Aplicação das regras de decisão
df_resultado = df_joined.withColumn("status_final",
    when(col("a.cpf").isNull(), "Q001")
    .when(col("score_final") >= 0.75, "OK")
    .when(col("score_final") < 0.60, "Q002")
    .when((col("a.data_nasc") != col("b.data_nasc")) & (col("a.sexo") != col("b.sexo")), "Q006")
    .otherwise("VALIDAR_NOMES")
)

# Seleção das colunas principais
df_resultado = df_resultado.select(
    col("a.id").alias("id_a"), col("b.id").alias("id_b"),
    col("a.cpf").alias("cpf"), col("score_final"), col("status_final")
)

# Dados prontos para ingestão e log de inconsistência
df_ingestao = df_resultado.filter(col("status_final") == "OK")
df_log = df_resultado.filter(col("status_final").startswith("Q"))

# Inclusão de registros únicos (sem duplicidade) diretamente na ingestão
cpfs_duplicados = df_resultado.select("cpf").distinct()
df_unicos = df.join(cpfs_duplicados, on="cpf", how="left_anti")
df_ingestao = df_ingestao.unionByName(df_unicos)

# Exemplo de salvamento
# df_ingestao.write.saveAsTable("curated.ingestao_final")
# df_log.write.saveAsTable("log.inconsistencias")
