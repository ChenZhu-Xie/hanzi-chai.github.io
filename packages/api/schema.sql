CREATE TABLE IF NOT EXISTS repertoire (
  unicode INTEGER PRIMARY KEY,
  tygf INTEGER NOT NULL,
  gb2312 INTEGER NOT NULL,
  glyphs TEXT NOT NULL,
  name TEXT,
  gf0014_id INTEGER,
  gf3001_id INTEGER,
  ambiguous INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS characters (
  `unicode` INTEGER PRIMARY KEY,
  `tygf` INTEGER,
  `gb2312` INTEGER,
  `glyphs` TEXT NOT NULL,
  `name` TEXT,
  `ambiguous` INTEGER
);

CREATE TABLE IF NOT EXISTS glyphs (
  `id` INTEGER PRIMARY KEY,
  `name` TEXT,
  `type` TEXT NOT NULL,
  `operator` TEXT,
  `references` TEXT,
  `strokes` TEXT,
  `gf0014_id` INTEGER,
  `gf3001_id` INTEGER,
  `ambiguous` INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
  `id` TEXT NOT NULL DEFAULT "" PRIMARY KEY,
  `name` TEXT NOT NULL DEFAULT "",
  `email` TEXT NOT NULL DEFAULT "",
  `password` TEXT NOT NULL DEFAULT "",
  `avatar` TEXT NOT NULL DEFAULT "",
  `role` INTEGER NOT NULL DEFAULT 0,
  `state` INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS equivalence (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `user` TEXT NOT NULL,
  `model` TEXT NOT NULL,
  `data` TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS glyph_relations (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT,
  `left_id` INTEGER NOT NULL,
  `right_id` INTEGER NOT NULL,
  `kind` TEXT NOT NULL,
  `status` TEXT NOT NULL,
  `provenance` TEXT NOT NULL,
  `sources` TEXT NOT NULL DEFAULT '[]',
  `evidence` TEXT NOT NULL DEFAULT '[]',
  CHECK (`left_id` < `right_id`),
  FOREIGN KEY (`left_id`) REFERENCES glyphs(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`right_id`) REFERENCES glyphs(`id`) ON DELETE CASCADE,
  UNIQUE (`left_id`, `right_id`, `kind`)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_name ON repertoire(name);
CREATE INDEX IF NOT EXISTS idx_glyph_relations_left ON glyph_relations(left_id);
CREATE INDEX IF NOT EXISTS idx_glyph_relations_right ON glyph_relations(right_id);
