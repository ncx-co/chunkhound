# Evaluation helpers (sane defaults, two targets):
#   make bench-lang      # semantic language eval on lang-eval-dev @ k=10
#   make bench-cluster   # clustering eval on cluster-stress-dev
#
# Optional: override config file via CONFIG, e.g.:
#   make bench-lang CONFIG=.chunkhound.json

.PHONY: bench-lang bench-cluster test-postgresql test-postgresql-up test-postgresql-down

bench-lang:
	uv run python -m chunkhound.tools.eval_search \
		--bench-id lang-eval-dev \
		--mode mixed \
		--search-mode semantic \
		--languages all \
		--k 10 \
		$(if $(CONFIG),--config $(CONFIG),) \
		--output .chunkhound/benches/lang-eval-dev/eval_semantic_k10.json

bench-cluster:
	uv run python -m chunkhound.tools.eval_cluster \
		--bench-id cluster-stress-dev \
		$(if $(CONFIG),--config $(CONFIG),) \
		--output .chunkhound/benches/cluster-stress-dev/cluster_eval.json

# PostgreSQL provider testing
test-postgresql:
	@./scripts/test-postgresql.sh

test-postgresql-up:
	@docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@timeout 60 sh -c 'until docker-compose -f docker-compose.test.yml exec -T postgres pg_isready -U chunkhound -d chunkhound_test 2>/dev/null; do sleep 1; done'
	@echo ""
	@echo "PostgreSQL is ready!"
	@echo "Connection: postgresql://chunkhound:chunkhound_test_pass@localhost:5433/chunkhound_test"

test-postgresql-down:
	@docker-compose -f docker-compose.test.yml down -v
