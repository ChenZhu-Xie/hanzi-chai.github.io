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

CREATE INDEX IF NOT EXISTS idx_glyph_relations_left ON glyph_relations(left_id);
CREATE INDEX IF NOT EXISTS idx_glyph_relations_right ON glyph_relations(right_id);
