#!/bin/bash
# HaQei v5 API テストスクリプト

API_URL="${1:-http://localhost:8787}"

echo "=== HaQei v5 API Test ==="
echo "API URL: $API_URL"
echo ""

# 1. 質問一覧取得
echo "1. GET /v5/diagnose/questions"
curl -s "$API_URL/v5/diagnose/questions" | jq '.version, .questions | length'
echo ""

# 2. プレビュー診断
echo "2. POST /v5/diagnose/preview"
curl -s -X POST "$API_URL/v5/diagnose/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "changeNature": {
        "expansion": 4,
        "contraction": 2,
        "maintenance": 2,
        "transformation": 2
      },
      "agency": 4,
      "timeframe": "shortTerm",
      "relationship": {
        "self": true,
        "family": false,
        "team": true,
        "organization": false,
        "external": false,
        "society": false
      },
      "emotionalTone": {
        "excitement": 4,
        "caution": 2,
        "anxiety": 1,
        "optimism": 3
      }
    }
  }' | jq '{resultId, topCandidate: .topCandidate.name, candidateCount, version}'
echo ""

# 3. 維持傾向のテスト
echo "3. POST /v5/diagnose/preview (維持傾向)"
curl -s -X POST "$API_URL/v5/diagnose/preview" \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "changeNature": {
        "expansion": 1,
        "contraction": 1,
        "maintenance": 5,
        "transformation": 1
      },
      "agency": 2,
      "timeframe": "midTerm",
      "relationship": {
        "self": false,
        "family": true,
        "team": true,
        "organization": false,
        "external": false,
        "society": false
      },
      "emotionalTone": {
        "excitement": 2,
        "caution": 4,
        "anxiety": 2,
        "optimism": 2
      }
    }
  }' | jq '{resultId, topCandidate: .topCandidate.name, candidateCount, version}'
echo ""

echo "=== Test Complete ==="
