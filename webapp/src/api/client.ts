import axios from "axios";

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:18080";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
});

