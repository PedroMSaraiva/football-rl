Coloque este repositório em uma pasta junto do repositório do google research football.

Sugerimos utilizar o fork: https://github.com/BrunoBSM/football

Ou seja, a estrutura da pasta deve ser a seguinte:
- rl-course-gfootball-helpers/
    - videos/
    - docker-compose.yml
    - test_recording.py
- football/
    - gfootball/
    - third_party/

## Sincronização de Arquivos com Docker

O `docker-compose.yml` está configurado para montar os arquivos do host no container, garantindo que as alterações sejam refletidas em tempo real. No entanto, se você modificar arquivos como `repro_scoring_easy_sb3.py` e notar que as mudanças não aparecem no container:

### Verificar Sincronização

Execute o script de verificação:
```bash
./check_sync.sh
```

Este script compara o hash MD5 do arquivo local com o do container para verificar se estão sincronizados.

### Forçar Sincronização

Se os arquivos não estiverem sincronizados, recrie o container:

```bash
docker-compose down
docker-compose up -d
```

### Estrutura de Volumes

O docker-compose monta:
- `../football` → `/workspace/football` (para desenvolvimento)
- `../football` → `/gfootball` (para sobrescrever arquivos copiados no build)
- `./` → `/RL` (para logs, checkpoints, etc.)

**Importante**: O Dockerfile copia os arquivos para `/gfootball` durante o build. O volume mount adicional garante que os arquivos sejam sempre atualizados, mas pode ser necessário recriar o container após mudanças significativas.


