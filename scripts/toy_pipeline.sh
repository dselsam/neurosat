echo ">> Generating data..."
bash scripts/toy_gen_data.sh
echo ">> Training..."
bash scripts/toy_train.sh
echo ">> Validating..."
bash scripts/toy_test.sh
echo ">> Solving..."
bash scripts/toy_solve.sh
