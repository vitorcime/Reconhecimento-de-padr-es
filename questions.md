Questões de pré-processamento
01. Nem sempre os canais são vistos como características. Uma outra forma é adicionar os canais às amostras (reduzindo a quantidade de características e aumentando a quantidade de amostras). O resultado disso deve ser avaliado.

Resultado avaliado.

02. É comum a aplicação de algum algoritmo para reduzir todos os canais ou transformar apenas em um (que é o caso de aplicar a média de todos os eletrodos/canais).

Sim, é comum. Como demonstrado, todos os vetores de características foram adicionados em uma lista única, referente as extrações da região analisada. Essa técnica é utilizada para reduzir e simplificar a análise do classificador.

03. Adicionar características ruins confundem o resultado? Características que não estão relacionadas ao domínio do problema pode ser ruim? Isso deve ser avaliado...

Como estamos analisando a região parietal, para-ociptal e ociptal (referentes a visão do indivíduo) podemos afirmar que ao adicionar características ruins o resultado pode ser afetado. Ao aplicarmos um cálculo de média em regiões que não processam informações da visão, ruídos podem ser considerados no resultado.
