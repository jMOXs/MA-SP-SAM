from ma_sp_sam.data.aimseg_index import records_from_archive_members


def test_records_aimseg_images_semantic_and_instance_members():
    members = [
        "Control_Dataset/Images/a.tif",
        "Control_Dataset/GroundTruth_Semantic/a.tif",
        "Control_Dataset/GroundTruth_Instance/a.tif",
        "Control_Dataset/Images/b.tif",
        "Control_Dataset/GroundTruth_Semantic/b.tif",
    ]

    records = records_from_archive_members("Control_Dataset.rar", members)

    assert len(records) == 2
    assert records[0].dataset == "Control_Dataset"
    assert records[0].sample_id == "a"
    assert records[0].image_member.endswith("Images/a.tif")
    assert records[0].semantic_member.endswith("GroundTruth_Semantic/a.tif")
    assert records[0].instance_member.endswith("GroundTruth_Instance/a.tif")
    assert records[1].instance_member is None
