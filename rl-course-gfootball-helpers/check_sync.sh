#!/bin/bash
# Script para verificar se os arquivos estão sincronizados entre host e container

echo "Verificando sincronização de arquivos entre host e container Docker..."
echo ""

CONTAINER_NAME="gfootball-dev"
LOCAL_FILE="../football/gfootball/repro_scoring_easy_sb3.py"
CONTAINER_FILE="/gfootball/gfootball/repro_scoring_easy_sb3.py"
CONTAINER_FILE_ALT="/workspace/football/gfootball/repro_scoring_easy_sb3.py"

# Verificar se o container está rodando
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "❌ Container $CONTAINER_NAME não está rodando!"
    echo "   Execute: docker-compose up -d"
    exit 1
fi

echo "✓ Container $CONTAINER_NAME está rodando"
echo ""

# Verificar se o arquivo local existe
if [ ! -f "$LOCAL_FILE" ]; then
    echo "❌ Arquivo local não encontrado: $LOCAL_FILE"
    exit 1
fi

echo "✓ Arquivo local encontrado: $LOCAL_FILE"
echo ""

# Calcular hash do arquivo local
LOCAL_HASH=$(md5sum "$LOCAL_FILE" 2>/dev/null | cut -d' ' -f1 || md5 -q "$LOCAL_FILE" 2>/dev/null)
if [ -z "$LOCAL_HASH" ]; then
    echo "⚠ Não foi possível calcular hash do arquivo local"
    LOCAL_HASH="N/A"
else
    echo "  Hash local: $LOCAL_HASH"
fi

# Verificar arquivo no container (primeiro em /gfootball)
if docker exec "$CONTAINER_NAME" test -f "$CONTAINER_FILE"; then
    echo "✓ Arquivo encontrado no container: $CONTAINER_FILE"
    CONTAINER_HASH=$(docker exec "$CONTAINER_NAME" md5sum "$CONTAINER_FILE" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$CONTAINER_HASH" ]; then
        CONTAINER_HASH=$(docker exec "$CONTAINER_NAME" md5 -q "$CONTAINER_FILE" 2>/dev/null)
    fi
    if [ -n "$CONTAINER_HASH" ]; then
        echo "  Hash no container: $CONTAINER_HASH"
        if [ "$LOCAL_HASH" = "$CONTAINER_HASH" ] && [ "$LOCAL_HASH" != "N/A" ]; then
            echo "  ✓ Arquivos estão sincronizados!"
        else
            echo "  ❌ Arquivos NÃO estão sincronizados!"
            echo ""
            echo "  Solução: Recrie o container para forçar sincronização:"
            echo "    docker-compose down"
            echo "    docker-compose up -d"
        fi
    fi
elif docker exec "$CONTAINER_NAME" test -f "$CONTAINER_FILE_ALT"; then
    echo "✓ Arquivo encontrado no container: $CONTAINER_FILE_ALT"
    CONTAINER_HASH=$(docker exec "$CONTAINER_NAME" md5sum "$CONTAINER_FILE_ALT" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$CONTAINER_HASH" ]; then
        CONTAINER_HASH=$(docker exec "$CONTAINER_NAME" md5 -q "$CONTAINER_FILE_ALT" 2>/dev/null)
    fi
    if [ -n "$CONTAINER_HASH" ]; then
        echo "  Hash no container: $CONTAINER_HASH"
        if [ "$LOCAL_HASH" = "$CONTAINER_HASH" ] && [ "$LOCAL_HASH" != "N/A" ]; then
            echo "  ✓ Arquivos estão sincronizados!"
        else
            echo "  ❌ Arquivos NÃO estão sincronizados!"
            echo ""
            echo "  Solução: Recrie o container para forçar sincronização:"
            echo "    docker-compose down"
            echo "    docker-compose up -d"
        fi
    fi
else
    echo "❌ Arquivo não encontrado no container em nenhum dos caminhos:"
    echo "   - $CONTAINER_FILE"
    echo "   - $CONTAINER_FILE_ALT"
fi

echo ""
echo "Para verificar manualmente, execute:"
echo "  docker exec $CONTAINER_NAME cat $CONTAINER_FILE | head -20"
echo "  head -20 $LOCAL_FILE"

