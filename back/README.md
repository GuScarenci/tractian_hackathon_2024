# Docker-Compose e Docker
- Para rodar o projeto completo
```shell
docker compose up
```
- Se for para rodar no background (não ficar vendo log da aplicação e segurando a sessão de shell)

```shell
docker compose up -d
```

- Caso o backend seja mudado, precisa dar build no container
```shell
docker compose build
```
- Se for para instalar bibliotecas dentro do container, editar o arquivo `Dockerfile` e então dar build


- Para resetar o backend completamente, tem que deletar todos os diretórios nas opções de "volumes" e final ":rw" dentro do docker-compose.yml com sudo


- Para parar de rodar tudo
```shell
docker compose down
```


- Para ver como os conaineres estão rodando (e se estão rodando ou não)
```shell
docker compose ps -a
```

- Dentro desse arquivo, tem vários serviços com nomes
```yaml
services:
  servico-1:
    ...
    ports:
      - "8080:8080"
    ...
  servico-2:
    ...
    ports:
      - "9000:9090"
    ...
  base-de-dados:
    ...
    ports:
      - "5432:5432"
    ...
```
Para ver dados de um container dentro de outro, usar o nome do serviço e a porta do ladoo **direito** dos dois pontos, para acessar localmente do seu PC, usar localhost e a porta do lado **esquerdo** dos dois pontos
- Dentro do container para acessar o servico-2 tem que usar "http://servico-2:9090"
- Fora do container para acessar o servico-2 tem que usar "http://localhost:9000"


- Se for para executar algo dentro do container **não recomendado** (normalmente o comando vai ser bash ou sh)
```shell
docker compose exec servico comando
```

- Especificamente numa base de dados postgres, para criar uma shell de SQL usar
```shell
docker exec -it postgres psql -U $POSTGRES_USES
```

# ngrok
Para expor um servidor em api:8080
```shell
docker run --net=back_default -it -e NGROK_AUTHTOKEN=$NGROK_AUTHTOKEN ngrok/ngrok:latest http http://api:8080
```
