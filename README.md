# Data-Analysis
Este é um repositório para análise de dados usando bibliotecas como numpy, pandas, matplotlib, scipy, e sklearn para realizar a análise e visualização dos dados.

O script BestDistFit realiza as seguintes funções principais:

#    Detecção da Melhor Distribuição:

        Para cada coluna do conjunto de dados, o script tenta ajustar diversas distribuições estatísticas (normais, log-normais, exponenciais, entre outras) e escolhe a distribuição com o melhor ajuste, baseado no erro quadrático (SSE) entre os dados observados e a distribuição teórica.

        O script também identifica variáveis binárias e categóricas (como variáveis de saída de modelos) e sugere a distribuição Bernoulli.

        Caso o ajuste paramétrico não seja adequado, o script recorre a uma estimativa de densidade kernel (KDE) para modelar a distribuição de dados.

#    Visualização das Distribuições Ajustadas:

        Para cada coluna, o script gera um gráfico de histograma dos dados, com a distribuição ajustada sobreposta.

        O gráfico também exibe os limites de confiança para os parâmetros da distribuição ajustada.

#    Análise de Dados:

        O script aplica a função de ajuste de distribuição para cada coluna do DataFrame e exibe os resultados, incluindo o tipo de distribuição sugerido, os parâmetros ajustados e a razão para a escolha.