export default {
  plugins: ['jest', '@typescript-eslint'],
  extends: [
    'plugin:github/recommended',
    'eslint:recommended',
    'plugin:prettier/recommended',
    'plugin:import/errors',
    'plugin:import/warnings',
    'plugin:import/typescript'
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: './tsconfig.json'
  },
  globals: {
    globalThis: false
  },
  rules: {
    'i18n-text/no-en': 'off'
  },
  env: {
    node: true,
    es6: true,
    'jest/globals': true
  },
  ignores: ['dist/', 'lib/', 'node_modules/', 'jest.config.js']
}
