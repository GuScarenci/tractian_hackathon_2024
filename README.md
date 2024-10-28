# Projeto: Prototipagem de Realidade Aumentada para Manutenção Assistida

## Visão Geral
Este projeto visa desenvolver uma aplicação de Realidade Aumentada (AR) para guiar técnicos de manutenção industrial de forma eficiente. A visão final é integrar um modelo Digital Twin com AR para monitoramento e diagnóstico em tempo real, possibilitando a identificação precisa de peças, ferramentas e instruções para manutenção. No entanto, para o hackathon atual, estamos focando em um protótipo básico usando Flutter com detecção de objetos simples via ResNet.

## Escopo do Protótipo
No estágio atual, implementamos um protótipo funcional em Flutter, onde a câmera do dispositivo identifica e destaca objetos usando ResNet, simulando a função de detecção de componentes das máquinas. Essa detecção de objetos é utilizada para fornecer informações básicas sobre o procedimento e as ferramentas necessárias para cada passo da manutenção.

**Telas do protótipo podem ser vistas abaixo**

![Tela de Login](images/1.png)
![Tela da Ordem](images/2.png)
![Tela de Passos](images/3.png)
![Tela de Passo 1](images/4.png)
![Tela de Passo 2](images/5.png)
![Tela de Passo 3](images/6.png)


## Tecnologias Utilizadas
- **Flutter**: Criação da interface do aplicativo e interação com o usuário.
- **ResNet**: Modelo de IA utilizado para detecção de objetos simples, simulando o reconhecimento de partes específicas nas máquinas.
- **AWS S3 (futuro)**: Armazenamento da base de dados de manuais para extração automatizada de informações usando IA e LLMs.
- **Digital Twin e AR (visão futura)**: Integração prevista para o futuro, com Digital Twin para mapeamento em tempo real e AR para guiar técnicos com superimposição de informações visuais.

## Funcionalidades do Protótipo

1. **Identificação de Objetos com ResNet**:
   - A aplicação captura uma imagem e utiliza o modelo ResNet para identificar componentes básicos, simulando a detecção de peças e ferramentas.

2. **Interface de AR Simples**:
   - Exibição de informações úteis sobre ferramentas e instruções de manutenção com base na detecção dos objetos.
   - A interface guia o técnico na manutenção através de uma lista de passos, com checkboxes para cada etapa concluída.

3. **Fluxo de Trabalho Passo a Passo**:
   - O técnico seleciona uma ordem de serviço e segue os passos indicados na interface.
   - Cada passo da ordem é marcado como concluído, e ao final, o técnico pode gerar um relatório básico do serviço realizado.

## Estrutura da API do Protótipo

1. **GET /ordem/<id>**
   - Retorna uma lista de passos para a ordem de serviço especificada pelo ID.
   - **Resposta**: JSON com array de passos.

2. **POST /ordem/<id>/imagem**
   - Envia uma imagem capturada para detecção de objetos e retorna a mesma imagem com bounding boxes simples (simulação de identificação de peças).
   - **Entrada**: Arquivo de imagem.
   - **Resposta**: Imagem com bounding boxes.

3. **POST /ordem**
   - Cria uma nova ordem de serviço.
   - **Entrada**: Dados sobre o problema e o ID da máquina.
   - **Resposta**: ID da nova ordem de serviço.

## Estrutura da Aplicação (Flutter)

1. **Login e Acesso de Empresa**:
   - Tela de login inicial para o técnico acessar o sistema.

2. **Lista de Ordens de Serviço**:
   - Exibe ordens de serviço disponíveis. O técnico seleciona uma para iniciar o processo de manutenção.

3. **Interface de AR Simples com Passo a Passo**:
   - Durante a manutenção, a câmera do dispositivo captura uma imagem do objeto, e o modelo ResNet faz uma detecção básica dos componentes, guiando o técnico no procedimento.

4. **Conclusão e Relatório Básico**:
   - Após concluir todos os passos, o técnico marca o serviço como concluído e envia um relatório resumido.

## Instalação e Configuração

1. **Requisitos de Ambiente**:
   - Flutter SDK

2. **Configuração do Projeto**:
   - Clone o repositório e instale as dependências do Flutter:
     ```bash
     flutter pub get
     ```

3. **Execução**:
   - Execute o aplicativo em um emulador ou dispositivo físico:
     ```bash
     flutter run
     ```

## Visão Futura: Digital Twin e AR com YOLO

Com o avanço do projeto, a meta é integrar:
- **Digital Twin** para fornecer um mapeamento detalhado da máquina em tempo real.
- **YOLO para AR**: Um modelo avançado de detecção de objetos para identificar com precisão cada peça e ferramenta, guiando o técnico visualmente.
- **Banco de Dados dos Manuais**: Armazenamento no AWS S3 com IA para extrair informações detalhadas de manutenção.

## Valor Agregado e Escalabilidade

1. **Escalabilidade com Digital Twin e YOLO (futuro)**:
   - Integração do Digital Twin permitirá a visualização precisa das máquinas em AR, com passos de manutenção personalizados para cada equipamento.
   - A modularidade do protótipo atual permite uma futura adaptação para múltiplos dispositivos, incluindo óculos AR.

2. **Integração com Tractian (futuro)**:
   - Utilizando dados da Tractian, a solução identificará causas raízes e guiará técnicos na resolução de problemas complexos de forma rápida e eficiente.

## Referências

- [Artigo MIT sobre Digital Twin e AR na Indústria](https://mitmetaverse.org/digital-twin/)
- [Frontiers: Uso do Apple Vision Pro para Manutenção Técnica](https://aerospacetechreview.com/klm-demonstrates-how-apple-vision-pro-can-improve-technical-maintenance/)
- [Digital Twin com YOLO: Práticas de Manutenção Virtual](https://www.frontiersin.org/journals/virtual-reality/articles/10.3389/frvir.2022.918685/full)
]