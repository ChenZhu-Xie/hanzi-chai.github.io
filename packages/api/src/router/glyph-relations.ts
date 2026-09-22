import { Router } from "itty-router";
import * as relations from "../controller/glyph-relations";

export const routerGlyphRelations = Router({ base: "/glyph-relations" })
  .get("/", relations.List)
  .post("/", relations.Create)
  .delete("/:id", relations.Delete);
